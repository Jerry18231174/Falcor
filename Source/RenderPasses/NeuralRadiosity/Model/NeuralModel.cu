#include "NeuralModel.h"
#include <cuda_runtime.h>
#include <type_traits>
#include <iostream>


namespace tcnn {

namespace {

template<typename SrcT, typename DstT>
void castMatrix(cudaStream_t stream, const GPUMatrixDynamic<SrcT>& src, GPUMatrixDynamic<DstT>& dst) {
    CHECK_THROW(src.m() == dst.m());
    CHECK_THROW(src.n() == dst.n());

    const uint32_t count = src.n_elements();
    parallel_for_gpu(stream, count, [srcData = src.data(), dstData = dst.data()] __device__ (size_t i) {
        dstData[i] = (DstT)srcData[i];
    });
}

template<typename T>
__global__ void clampGradients(
    T* gradients,
    uint32_t count,
    float threshold
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    float g = (float)gradients[idx];
    if (!isfinite(g)) {
        gradients[idx] = (T)0.0f;
        return;
    }

    gradients[idx] = (T)fminf(fmaxf(g, -threshold), threshold);
}

} // namespace

template<typename T>
NeuralModel<T>::NeuralModel(HashGrid::Config gridConfig) {
    mGridConfig = gridConfig;

    encOffset = 0;
    posOffset = encOffset + mGridConfig.nLevels * mGridConfig.nFeaturesPerLevel;
    dirOffset = posOffset + 3;
    normalOffset = dirOffset + 3;
    albedoOffset = normalOffset + 3;
    roughnessOffset = albedoOffset + 3;
    totalDim = roughnessOffset + 1;
    mInputDim = padUp(totalDim, 16);

    CHECK_THROW(mMaxBatchSize % 256 == 0);

    const json networkConfig = {
        {"otype", "FullyFusedMLP"},
        {"activation", "ReLU"},
        {"output_activation", "SquarePlus"},
        {"n_input_dims", mInputDim},
        {"n_output_dims", mOutputDim},
        {"n_neurons", 64},
        {"n_hidden_layers", 3}
    };
    mpNet = std::shared_ptr<Network<T, T>>(create_network<T>(networkConfig));

    {
        HashGrid::initializeConstants(mGridConfig);

        float resolution = mGridConfig.baseResolution;
        uint32_t gridSize = 0;
        for (int i = 0; i < mGridConfig.nLevels; i++)
        {
            uint64_t elems64 = (uint64_t) resolution + 1;
            elems64 = elems64 * elems64 * elems64;
            uint32_t elems = (uint32_t) std::min(elems64, (uint64_t) (1ull << mGridConfig.log2HashMapSize));

            resolution *= mGridConfig.perLevelScale;
            gridSize += elems;
        }
        mGridSize = gridSize * mGridConfig.nFeaturesPerLevel;
    }
}

template<typename T>
void NeuralModel<T>::inference_mixed_precision_impl(
    cudaStream_t stream,
    const GPUMatrixDynamic<float>& input,
    GPUMatrixDynamic<T>& output,
    bool use_inference_params
) {
    const uint32_t batchSize = input.n();
    CHECK_THROW(batchSize <= mMaxBatchSize);
    GPUMatrix<T> networkInput{mInputDim, batchSize, stream};
    castMatrix(stream, input, networkInput);

    HashGrid::launchForward(
        stream,
        mpGridsInference,
        networkInput.data(),
        batchSize,
        mInputDim,
        encOffset,
        posOffset,
        mGridConfig
    );

    mpNet->inference_mixed_precision(stream, networkInput, output, use_inference_params);
}

template<typename T>
std::unique_ptr<Context> NeuralModel<T>::forward_impl(
    cudaStream_t stream,
    const GPUMatrixDynamic<float>& input,
    GPUMatrixDynamic<T>* output,
    bool use_inference_params,
    bool prepare_input_gradients
) {
    const uint32_t batchSize = input.n();
    CHECK_THROW(batchSize <= mMaxBatchSize);

    auto ctx = std::make_unique<NeuralModelContext<T>>();
    ctx->batchSize = batchSize;
    ctx->networkInput = GPUMatrix<T>{mInputDim, batchSize, stream};
    castMatrix(stream, input, ctx->networkInput);

    HashGrid::launchForward(
        stream,
        mpGrids,
        ctx->networkInput.data(),
        batchSize,
        mInputDim,
        encOffset,
        posOffset,
        mGridConfig
    );

    ctx->netCtx = mpNet->forward(stream, ctx->networkInput, output, use_inference_params, prepare_input_gradients);

    return ctx;
}

template<typename T>
void NeuralModel<T>::backward_impl(
    cudaStream_t stream,
    const Context& ctx,
    const GPUMatrixDynamic<float>& input,
    const GPUMatrixDynamic<T>& output,
    const GPUMatrixDynamic<T>& dL_doutput,
    GPUMatrixDynamic<float>* dL_dinput,
    bool use_inference_params,
    GradientMode param_gradients_mode
) {
    const uint32_t batchSize = input.n();
    CHECK_THROW(batchSize <= mMaxBatchSize);
    
    const NeuralModelContext<T>& nCtx = static_cast<const NeuralModelContext<T>&>(ctx);
    GPUMatrixDynamic<T> dL_dnetwork_input{mInputDim, batchSize, stream};
    if (param_gradients_mode == GradientMode::Overwrite) {
        CUDA_CHECK_THROW(cudaMemsetAsync(mpGridsGradient, 0, sizeof(T) * mGridSize, stream));
    }

    mpNet->backward(stream, *nCtx.netCtx, nCtx.networkInput, output, dL_doutput, &dL_dnetwork_input, use_inference_params, param_gradients_mode);

    HashGrid::launchBackward(
        stream,
        mpGrids,
        nCtx.networkInput.data(),
        dL_dnetwork_input.data(),
        mpGridsGradient,
        batchSize,
        mInputDim,
        encOffset,
        posOffset,
        mGridConfig
    );

    if (dL_dinput) {
        castMatrix(stream, dL_dnetwork_input, *dL_dinput);
    }

    // Clamp gradients to prevent training instability.
    {
        const uint32_t count = n_params();
        const uint32_t blockSize = 256;
        const uint32_t gridSize = (count + blockSize - 1) / blockSize;
        clampGradients<<<gridSize, blockSize, 0, stream>>>(this->gradients(), count, 1.0f);
    }
}


template<typename T>
uint32_t NeuralModel<T>::input_width() const {
    return mInputDim;
}

template<typename T>
uint32_t NeuralModel<T>::padded_output_width() const {
    return padUp(output_width(), 16);
}

template<typename T>
uint32_t NeuralModel<T>::output_width() const {
    return mOutputDim;
}

template<typename T>
uint32_t NeuralModel<T>::required_input_alignment() const {
    return 16;
}

template<typename T>
json NeuralModel<T>::hyperparams() const {
    return {
        {"otype", "NeuralModel"},
    };
}

template<typename T>
void NeuralModel<T>::set_params_impl(T* params, T* inference_params, T* gradients) {
    uint32_t gridsOffset = 0;
    uint32_t netOffset = gridsOffset + padUp(mGridSize, 16);

    mpGrids = params + gridsOffset;
    mpGridsInference = inference_params + gridsOffset;
    mpGridsGradient = gradients + gridsOffset;
    mpNet->set_params(params + netOffset, inference_params + netOffset, gradients + netOffset);
}

template<typename T>
void NeuralModel<T>::initialize_params(pcg32& rnd, float* params_full_precision, float scale) {
    CUDA_CHECK_THROW(cudaMemsetAsync(params_full_precision, 0, sizeof(float) * padUp(mGridSize, 16)));
    mpNet->initialize_params(rnd, params_full_precision + padUp(mGridSize, 16), scale);
}

template<typename T>
size_t NeuralModel<T>::n_params() const {
    return padUp(mGridSize, 16) + mpNet->n_params();
}

template<typename T>
std::vector<std::pair<uint32_t, uint32_t>> NeuralModel<T>::layer_sizes() const {
    return {};
}

template class NeuralModel<precision_t>;


}
