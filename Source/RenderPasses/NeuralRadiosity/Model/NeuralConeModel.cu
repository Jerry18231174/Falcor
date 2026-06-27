#include "NeuralConeModel.h"
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
NeuralConeModel<T>::NeuralConeModel(HashGrid::Config primGridConfig, HashGridInterp::Config clsGridConfig) {
    mPrimGridConfig = primGridConfig;
    mClsGridConfig = clsGridConfig;

    primEncOffset = 0;
    clsEncOffset = primEncOffset + mPrimGridConfig.nLevels * mPrimGridConfig.nFeaturesPerLevel;
    posOffset = clsEncOffset + mClsGridConfig.nClusters * mClsGridConfig.nFeaturesPerLevel;
    dirOffset = posOffset + 3;
    normalOffset = dirOffset + 3;
    albedoOffset = normalOffset + 3;
    roughnessOffset = albedoOffset + 3;
    clsPosOffset = roughnessOffset + 1;
    clsScaleOffset = clsPosOffset + mClsGridConfig.nClusters * 3;
    clsWeightOffset = clsScaleOffset + mClsGridConfig.nClusters;
    totalDim = clsWeightOffset + mClsGridConfig.nClusters;
    mInputDim = padUp(totalDim, 16);

    CHECK_THROW(mMaxBatchSize % 256 == 0);

    const json networkConfig = {
        {"otype", "FullyFusedMLP"},
        {"activation", "ReLU"},
        {"output_activation", "SquarePlus"},
        {"n_input_dims", mInputDim},
        {"n_output_dims", mOutputDim},
        {"n_neurons", 64},
        {"n_hidden_layers", 4}
    };
    mpNet = std::shared_ptr<Network<T, T>>(create_network<T>(networkConfig));

    {
        HashGrid::initializeConstants(mPrimGridConfig);

        float resolution = mPrimGridConfig.baseResolution;
        uint32_t gridSize = 0;
        for (int i = 0; i < mPrimGridConfig.nLevels; i++)
        {
            uint64_t elems64 = (uint64_t) resolution + 1;
            elems64 = elems64 * elems64 * elems64;
            uint32_t elems = (uint32_t) std::min(elems64, (uint64_t) (1ull << mPrimGridConfig.log2HashMapSize));

            resolution *= mPrimGridConfig.perLevelScale;
            gridSize += elems;
        }
        mPrimGridSize = gridSize * mPrimGridConfig.nFeaturesPerLevel;
    }

    {
        HashGridInterp::initializeConstants(mClsGridConfig);

        float resolution = mClsGridConfig.baseResolution;
        uint32_t gridSize = 0;
        for (int i = 0; i < mClsGridConfig.nLevels; i++)
        {
            uint64_t elems64 = (uint64_t) resolution + 1;
            elems64 = elems64 * elems64 * elems64;
            uint32_t elems = (uint32_t) std::min(elems64, (uint64_t) (1ull << mClsGridConfig.log2HashMapSize));

            resolution *= mClsGridConfig.perLevelScale;
            gridSize += elems;
        }
        mClsGridSize = gridSize * mClsGridConfig.nFeaturesPerLevel;
    }
}

template<typename T>
void NeuralConeModel<T>::inference_mixed_precision_impl(
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
        mpPrimGridsInference,
        networkInput.data(),
        batchSize,
        mInputDim,
        primEncOffset,
        posOffset,
        mPrimGridConfig
    );

    HashGridInterp::launchForward(
        stream,
        mpClsGridsInference,
        networkInput.data(),
        batchSize,
        mInputDim,
        clsEncOffset,
        clsPosOffset,
        clsScaleOffset,
        clsWeightOffset,
        mClsGridConfig
    );

    mpNet->inference_mixed_precision(stream, networkInput, output, use_inference_params);
}

template<typename T>
std::unique_ptr<Context> NeuralConeModel<T>::forward_impl(
    cudaStream_t stream,
    const GPUMatrixDynamic<float>& input,
    GPUMatrixDynamic<T>* output,
    bool use_inference_params,
    bool prepare_input_gradients
) {
    const uint32_t batchSize = input.n();
    CHECK_THROW(batchSize <= mMaxBatchSize);

    auto ctx = std::make_unique<NeuralConeModelContext<T>>();
    ctx->batchSize = batchSize;
    ctx->networkInput = GPUMatrix<T>{mInputDim, batchSize, stream};
    castMatrix(stream, input, ctx->networkInput);

    HashGrid::launchForward(
        stream,
        mpPrimGrids,
        ctx->networkInput.data(),
        batchSize,
        mInputDim,
        primEncOffset,
        posOffset,
        mPrimGridConfig
    );

    HashGridInterp::launchForward(
        stream,
        mpClsGrids,
        ctx->networkInput.data(),
        batchSize,
        mInputDim,
        clsEncOffset,
        clsPosOffset,
        clsScaleOffset,
        clsWeightOffset,
        mClsGridConfig
    );

    ctx->netCtx = mpNet->forward(stream, ctx->networkInput, output, use_inference_params, prepare_input_gradients);

    return ctx;
}

template<typename T>
void NeuralConeModel<T>::backward_impl(
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

    const NeuralConeModelContext<T>& ncCtx = static_cast<const NeuralConeModelContext<T>&>(ctx);
    GPUMatrixDynamic<T> dL_dnetwork_input{mInputDim, batchSize, stream};
    if (param_gradients_mode == GradientMode::Overwrite) {
        CUDA_CHECK_THROW(cudaMemsetAsync(mpPrimGridsGradient, 0, sizeof(T) * mPrimGridSize, stream));
        CUDA_CHECK_THROW(cudaMemsetAsync(mpClsGridsGradient, 0, sizeof(T) * mClsGridSize, stream));
    }

    mpNet->backward(stream, *ncCtx.netCtx, ncCtx.networkInput, output, dL_doutput, &dL_dnetwork_input, use_inference_params, param_gradients_mode);

    HashGrid::launchBackward(
        stream,
        mpPrimGrids,
        ncCtx.networkInput.data(),
        dL_dnetwork_input.data(),
        mpPrimGridsGradient,
        batchSize,
        mInputDim,
        primEncOffset,
        posOffset,
        mPrimGridConfig
    );

    HashGridInterp::launchBackward(
        stream,
        mpClsGrids,
        ncCtx.networkInput.data(),
        dL_dnetwork_input.data(),
        mpClsGridsGradient,
        batchSize,
        mInputDim,
        clsEncOffset,
        clsPosOffset,
        clsScaleOffset,
        clsWeightOffset,
        mClsGridConfig
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
uint32_t NeuralConeModel<T>::input_width() const {
    return mInputDim;
}

template<typename T>
uint32_t NeuralConeModel<T>::padded_output_width() const {
    return padUp(output_width(), 16);
}

template<typename T>
uint32_t NeuralConeModel<T>::output_width() const {
    return mOutputDim;
}

template<typename T>
uint32_t NeuralConeModel<T>::required_input_alignment() const {
    return 16;
}

template<typename T>
json NeuralConeModel<T>::hyperparams() const {
    return {
        {"otype", "NeuralConeModel"},
    };
}

template<typename T>
void NeuralConeModel<T>::set_params_impl(T* params, T* inference_params, T* gradients) {
    uint32_t primGridsOffset = 0;
    uint32_t clsGridOffset = primGridsOffset + padUp(mPrimGridSize, 16);
    uint32_t netOffset = clsGridOffset + padUp(mClsGridSize, 16);

    mpPrimGrids = params + primGridsOffset;
    mpPrimGridsInference = inference_params + primGridsOffset;
    mpPrimGridsGradient = gradients + primGridsOffset;
    mpClsGrids = params + clsGridOffset;
    mpClsGridsInference = inference_params + clsGridOffset;
    mpClsGridsGradient = gradients + clsGridOffset;
    mpNet->set_params(params + netOffset, inference_params + netOffset, gradients + netOffset);
}

template<typename T>
void NeuralConeModel<T>::initialize_params(pcg32& rnd, float* params_full_precision, float scale) {
    CUDA_CHECK_THROW(cudaMemsetAsync(params_full_precision, 0, sizeof(float) * (padUp(mPrimGridSize, 16) + padUp(mClsGridSize, 16))));
    uint32_t netOffset = padUp(mPrimGridSize, 16) + padUp(mClsGridSize, 16);
    mpNet->initialize_params(rnd, params_full_precision + netOffset, scale);
}

template<typename T>
size_t NeuralConeModel<T>::n_params() const {
    return padUp(mPrimGridSize, 16) + padUp(mClsGridSize, 16) + mpNet->n_params();
}

template<typename T>
std::vector<std::pair<uint32_t, uint32_t>> NeuralConeModel<T>::layer_sizes() const {
    return {};
}

template class NeuralConeModel<precision_t>;


}
