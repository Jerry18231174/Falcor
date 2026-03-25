#include "NeuralModel.h"
#include <cuda_runtime.h>
#include <type_traits>
#include <iostream>


namespace tcnn {

namespace {

__global__ void clampGradients(
    float* gradients,
    uint32_t count,
    float threshold
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    float g = gradients[idx];
    if (!isfinite(g)) {
        gradients[idx] = 0.0f;
        return;
    }

    gradients[idx] = fminf(fmaxf(g, -threshold), threshold);
}

} // namespace

template<typename T>
NeuralModel<T>::NeuralModel() {
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
    mpNet = std::shared_ptr<Network<float, float>>(create_network<float>(networkConfig));

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
    const GPUMatrixDynamic<T>& input,
    GPUMatrixDynamic<T>& output,
    bool use_inference_params
) {
    static_assert(std::is_same<T, float>::value, "NeuralModel currently supports float inference only.");

    const uint32_t batchSize = input.n();
    CHECK_THROW(batchSize <= mMaxBatchSize);

    HashGrid::launchForward(
        stream,
        mpGridsInference,
        input.data(),
        batchSize,
        mInputDim,
        encOffset,
        posOffset,
        mGridConfig
    );

    mpNet->inference(stream, input, output, use_inference_params);
}

template<typename T>
std::unique_ptr<Context> NeuralModel<T>::forward_impl(
    cudaStream_t stream,
    const GPUMatrixDynamic<T>& input,
    GPUMatrixDynamic<T>* output,
    bool use_inference_params,
    bool prepare_input_gradients
) {
    static_assert(std::is_same<T, float>::value, "NeuralModel currently supports float inference only.");

    const uint32_t batchSize = input.n();
    CHECK_THROW(batchSize <= mMaxBatchSize);

    auto ctx = std::make_unique<NeuralModelContext<T>>();
    ctx->batchSize = batchSize;

    HashGrid::launchForward(
        stream,
        mpGrids,
        input.data(),
        batchSize,
        mInputDim,
        encOffset,
        posOffset,
        mGridConfig
    );

    ctx->netCtx = mpNet->forward(stream, input, output, use_inference_params, prepare_input_gradients);

    return ctx;
}

template<typename T>
void NeuralModel<T>::backward_impl(
    cudaStream_t stream,
    const Context& ctx,
    const GPUMatrixDynamic<T>& input,
    const GPUMatrixDynamic<T>& output,
    const GPUMatrixDynamic<T>& dL_doutput,
    GPUMatrixDynamic<T>* dL_dinput,
    bool use_inference_params,
    GradientMode param_gradients_mode
) {
    static_assert(std::is_same<T, float>::value, "NeuralModel currently supports float inference only.");

    const uint32_t batchSize = input.n();
    CHECK_THROW(batchSize <= mMaxBatchSize);
    
    const NeuralModelContext<T>& nCtx = static_cast<const NeuralModelContext<T>&>(ctx);
    if (param_gradients_mode == GradientMode::Overwrite) {
        CUDA_CHECK_THROW(cudaMemsetAsync(mpGridsGradient, 0, sizeof(T) * mGridSize, stream));
    }

    mpNet->backward(stream, *nCtx.netCtx, input, output, dL_doutput, dL_dinput, use_inference_params, param_gradients_mode);

    HashGrid::launchBackward(
        stream,
        mpGrids,
        input.data(),
        dL_dinput->data(),
        mpGridsGradient,
        batchSize,
        mInputDim,
        encOffset,
        posOffset,
        mGridConfig
    );

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
    uint32_t netOffset = gridsOffset + mGridSize;

    mpGrids = params + gridsOffset;
    mpGridsInference = inference_params + gridsOffset;
    mpGridsGradient = gradients + gridsOffset;
    mpNet->set_params(params + netOffset, inference_params + netOffset, gradients + netOffset);
}

template<typename T>
void NeuralModel<T>::initialize_params(pcg32& rnd, float* params_full_precision, float scale) {
    CUDA_CHECK_THROW(cudaMemsetAsync(params_full_precision, 0, sizeof(T) * mGridSize));
    mpNet->initialize_params(rnd, params_full_precision + mGridSize, scale);
}

template<typename T>
size_t NeuralModel<T>::n_params() const {
    return mGridSize + mpNet->n_params();
}

template<typename T>
std::vector<std::pair<uint32_t, uint32_t>> NeuralModel<T>::layer_sizes() const {
    return {};
}

template class NeuralModel<float>;


}
