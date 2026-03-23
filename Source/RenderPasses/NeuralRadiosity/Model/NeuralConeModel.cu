#include "NeuralConeModel.h"
#include "HashGridInterp.h"
#include <cuda_runtime.h>
#include <type_traits>
#include <iostream>


namespace tcnn {

uint32_t padUp(uint32_t x, uint32_t align) {
    return (x + align - 1) & ~(align - 1);
}

__global__ void constructMergeInputKernel(
    const float* specPrimInput,
    const float* specPrimOutput,
    const float* clsInput,
    const float* clsOutput,
    float* mergeInput,
    uint32_t batchSize,
    uint32_t numClusters,
    uint32_t hashFeatureDim,
    uint32_t primInputDim,
    uint32_t clsInputDim,
    uint32_t mergeInputDim,
    uint32_t outputDim
) {
    uint32_t pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixelIdx >= batchSize) {
        return;
    }

    const float* primInput = specPrimInput + pixelIdx * primInputDim;
    const float* primOutput = specPrimOutput + pixelIdx * outputDim;
    float* mergeIn = mergeInput + pixelIdx * mergeInputDim;

    // Diffuse color.
    mergeIn[0] = primOutput[0];
    mergeIn[1] = primOutput[1];
    mergeIn[2] = primOutput[2];
    mergeIn[3] = primOutput[3];
    
    // Specular color: weighted average of cluster outputs.
    float weighted[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t c = 0; c < numClusters; ++c) {
        uint32_t clusterIdx = pixelIdx * numClusters + c;
        const float w = clsInput[clusterIdx * clsInputDim + hashFeatureDim + 7];
        const float* clsOut = clsOutput + clusterIdx * outputDim;
        weighted[0] += w * clsOut[0];
        weighted[1] += w * clsOut[1];
        weighted[2] += w * clsOut[2];
        weighted[3] += w * clsOut[3];
    }
    mergeIn[4] = weighted[0];
    mergeIn[5] = weighted[1];
    mergeIn[6] = weighted[2];
    mergeIn[7] = weighted[3];

    // Direction.
    mergeIn[8] = primInput[6];
    mergeIn[9] = primInput[7];
    mergeIn[10] = primInput[8];

    // Albedo & roughness.
    mergeIn[11] = primInput[12];
    mergeIn[12] = primInput[13];
    mergeIn[13] = primInput[14];
    mergeIn[14] = primInput[15];

    for (uint32_t i = 15; i < mergeInputDim; ++i) {
        mergeIn[i] = 0.0f;
    }
}

__global__ void constructdLdClsOutputKernel(
    // const float* specPrimInput,
    // const float* specPrimOutput,
    const float* clsInput,
    // const float* clsOutput,
    // const float* mergeInput,
    const float* dL_dmergeInput,
    float* dL_dclsOutput,
    float* dL_dspecPrimOutput,
    uint32_t batchSize,
    uint32_t numClusters,
    uint32_t hashFeatureDim,
    // uint32_t primInputDim,
    uint32_t clsInputDim,
    uint32_t mergeInputDim,
    uint32_t outputDim
) {
    uint32_t pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixelIdx >= batchSize) {
        return;
    }

    // const float* primInput = specPrimInput + pixelIdx * primInputDim;
    // const float* primOutput = specPrimOutput + pixelIdx * outputDim;
    // const float* mergeIn = mergeInput + pixelIdx * mergeInputDim;
    const float* dL_dmergeIn = dL_dmergeInput + pixelIdx * mergeInputDim;
    float* dL_dprimOutput = dL_dspecPrimOutput + pixelIdx * outputDim;

    dL_dprimOutput[0] = dL_dmergeIn[0];
    dL_dprimOutput[1] = dL_dmergeIn[1];
    dL_dprimOutput[2] = dL_dmergeIn[2];
    dL_dprimOutput[3] = dL_dmergeIn[3];

    for (uint32_t i = 4; i < outputDim; ++i) {
        dL_dprimOutput[i] = 0.0f;
    }

    for (uint32_t c = 0; c < numClusters; ++c) {
        uint32_t clusterIdx = pixelIdx * numClusters + c;
        const float w = clsInput[clusterIdx * clsInputDim + hashFeatureDim + 7];
        float* dL_dclsOut = dL_dclsOutput + clusterIdx * outputDim;
        dL_dclsOut[0] = w * dL_dmergeIn[4];
        dL_dclsOut[1] = w * dL_dmergeIn[5];
        dL_dclsOut[2] = w * dL_dmergeIn[6];
        dL_dclsOut[3] = w * dL_dmergeIn[7];

        for (uint32_t i = 4; i < outputDim; ++i) {
            dL_dclsOut[i] = 0.0f;
        }
    }
}

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

template<typename T>
NeuralConeModel<T>::NeuralConeModel() {
    mClsInputDim = padUp(mClsInputDim, 16);
    mMergeInputDim = padUp(mMergeInputDim, 16);
    mOutputDim = padUp(mOutputDim, 16);
    CHECK_THROW(mMaxBatchSize % 256 == 0);

    HashGridInterp::initializeConstants(
        nLevels,
        nFeaturesPerLevel,
        log2HashMapSize,
        baseResolution,
        perLevelScale,
        interpRatio
    );

    const tcnn::json diffEncodingConfig = {
        {"otype", "Composite"},
        {"nested", {
            {
                {"otype", "HashGrid"},
                {"n_dims_to_encode", 3},
                {"n_levels", nLevels},
                {"n_features_per_level", nFeaturesPerLevel},
                {"log2_hashmap_size", log2HashMapSize},
                {"base_resolution", baseResolution},
                {"per_level_scale", perLevelScale},
                {"interpolation", "Linear"}
            }
        }}
    };

    const tcnn::json diffNetConfig = {
        {"otype", "FullyFusedMLP"},
        {"activation", "ReLU"},
        {"output_activation", "SquarePlus"},
        {"n_neurons", 128},
        {"n_hidden_layers", 3}
    };

    mpPrimNet = std::make_shared<tcnn::NetworkWithInputEncoding<float>>(mPrimInputDim, mOutputDim, diffEncodingConfig, diffNetConfig);

    const json specNetConfig = {
        {"otype", "FullyFusedMLP"},
        {"activation", "ReLU"},
        {"output_activation", "SquarePlus"},
        {"n_input_dims", mClsInputDim},
        {"n_output_dims", mOutputDim},
        {"n_neurons", 64},
        {"n_hidden_layers", 2}
    };

    const json mergeNetConfig = {
        {"otype", "FullyFusedMLP"},
        {"activation", "ReLU"},
        {"output_activation", "SquarePlus"},
        {"n_input_dims", mMergeInputDim},
        {"n_output_dims", mOutputDim},
        {"n_neurons", 32},
        {"n_hidden_layers", 1}
    };

    float resolution = baseResolution;
    uint32_t gridSize = 0;
    for (int i = 0; i < nLevels; i++)
    {
        uint64_t elems64 = (uint64_t) resolution + 1;
        elems64 = elems64 * elems64 * elems64;
        uint32_t elems = (uint32_t) std::min(elems64, (uint64_t) (1ull << log2HashMapSize));

        resolution *= perLevelScale;
        gridSize += elems;
    }

    mGridSize = gridSize * nFeaturesPerLevel;
    mpSpecNet = std::shared_ptr<Network<float, float>>(create_network<float>(specNetConfig));
    mpMergeNet = std::shared_ptr<Network<float, float>>(create_network<float>(mergeNetConfig));

    const uint32_t maxClsBatchSize = mMaxBatchSize * numClusters;
    mpDiffPrimOutput = std::make_unique<GPUMemory<float>>(mOutputDim * mMaxBatchSize);
    mpSpecPrimOutput = std::make_unique<GPUMemory<float>>(mOutputDim * mMaxBatchSize);
    mpdLdSpecPrimOutput = std::make_unique<GPUMemory<float>>(mOutputDim * mMaxBatchSize);
    // mpClsInput = std::make_unique<GPUMemory<float>>(mClsInputDim * maxClsBatchSize);
    mpdLdClsInput = std::make_unique<GPUMemory<float>>(mClsInputDim * maxClsBatchSize);
    mpClsOutput = std::make_unique<GPUMemory<float>>(mOutputDim * maxClsBatchSize);
    mpdLdClsOutput = std::make_unique<GPUMemory<float>>(mOutputDim * maxClsBatchSize);
    mpMergeInput = std::make_unique<GPUMemory<float>>(mMergeInputDim * mMaxBatchSize);
    mpdLdMergeInput = std::make_unique<GPUMemory<float>>(mMergeInputDim * mMaxBatchSize);
}

template<typename T>
void NeuralConeModel<T>::inference_mixed_precision_impl(
    cudaStream_t stream,
    const GPUMatrixDynamic<T>& input,
    GPUMatrixDynamic<T>& output,
    bool use_inference_params
) {
    static_assert(std::is_same<T, float>::value, "NeuralConeModel currently supports float inference only.");

    const uint32_t batchSize = input.n();
    CHECK_THROW(batchSize <= mMaxBatchSize);
    const uint32_t diffBatchSize = mpDiffPrimInput->n();
    const uint32_t specBatchSize = mpSpecPrimInput->n();
    const uint32_t clsBatchSize = specBatchSize * numClusters;

    if (diffBatchSize > 0) {
        // Primary network of diffuse branch.
        GPUMatrix<float> diffPrimInput(mpDiffPrimInput->data(), mPrimInputDim, diffBatchSize);
        GPUMatrix<float> diffPrimOutput(output.data(), mOutputDim, diffBatchSize);
        mpPrimNet->inference(stream, diffPrimInput, diffPrimOutput, use_inference_params);
    }

    if (specBatchSize == 0) {
        return;
    }

    // Primary network of specular branch.
    GPUMatrix<float> specPrimInput(mpSpecPrimInput->data(), mPrimInputDim, specBatchSize);
    GPUMatrix<float> specPrimOutput(mpSpecPrimOutput->data(), mOutputDim, specBatchSize);
    mpPrimNet->inference(stream, specPrimInput, specPrimOutput, use_inference_params);

    // Specular network of specular branch.
    GPUMatrix<float> clsInput(mpClsInput->data(), mClsInputDim, clsBatchSize);
    GPUMatrix<float> clsOutput(mpClsOutput->data(), mOutputDim, clsBatchSize);
    GPUMatrix<float> mergeInput(mpMergeInput->data(), mMergeInputDim, specBatchSize);
    GPUMatrix<float> mergeOutput(output.data() + diffBatchSize * mOutputDim, mOutputDim, specBatchSize);

    {
        HashGridInterp::launchForward(
            stream,
            mpSpecGridsInference,
            clsInput.data(),
            clsBatchSize,
            mClsInputDim,
            nLevels,
            nFeaturesPerLevel,
            log2HashMapSize,
            baseResolution,
            perLevelScale,
            interpRatio
        );
    }

    mpSpecNet->inference(stream, clsInput, clsOutput, use_inference_params);

    {
        const uint32_t count = specBatchSize;
        const uint32_t blockSize = 256;
        const uint32_t gridSize = (count + blockSize - 1) / blockSize;
        constructMergeInputKernel<<<gridSize, blockSize, 0, stream>>>(
            specPrimInput.data(),
            specPrimOutput.data(),
            clsInput.data(),
            clsOutput.data(),
            mergeInput.data(),
            count,
            numClusters,
            nFeaturesPerLevel,
            mPrimInputDim,
            mClsInputDim,
            mMergeInputDim,
            mOutputDim
        );
    }

    mpMergeNet->inference(stream, mergeInput, mergeOutput, use_inference_params);
}

template<typename T>
std::unique_ptr<Context> NeuralConeModel<T>::forward_impl(
    cudaStream_t stream,
    const GPUMatrixDynamic<T>& input,
    GPUMatrixDynamic<T>* output,
    bool use_inference_params,
    bool prepare_input_gradients
) {
    static_assert(std::is_same<T, float>::value, "NeuralConeModel currently supports float inference only.");

    const uint32_t batchSize = input.n();
    CHECK_THROW(batchSize <= mMaxBatchSize);
    const uint32_t diffBatchSize = mpDiffPrimInput->n();
    const uint32_t specBatchSize = mpSpecPrimInput->n();
    const uint32_t clsBatchSize = specBatchSize * numClusters;

    auto ctx = std::make_unique<NeuralConeModelContext<T>>();
    ctx->diffBatchSize = diffBatchSize;
    ctx->specBatchSize = specBatchSize;
    ctx->clsBatchSize = clsBatchSize;

    if (diffBatchSize > 0) {
        // Primary network of diffuse branch.
        GPUMatrix<float> diffPrimInput(mpDiffPrimInput->data(), mPrimInputDim, diffBatchSize);
        GPUMatrix<float> diffPrimOutput(output->data(), mOutputDim, diffBatchSize);
        ctx->diffPrimCtx = mpPrimNet->forward(stream, diffPrimInput, &diffPrimOutput, use_inference_params, prepare_input_gradients);
    }

    if (specBatchSize == 0) {
        return ctx;
    }

    // Primary network of specular branch.
    GPUMatrix<float> specPrimInput(mpSpecPrimInput->data(), mPrimInputDim, specBatchSize);
    GPUMatrix<float> specPrimOutput(mpSpecPrimOutput->data(), mOutputDim, specBatchSize);
    ctx->specPrimCtx = mpPrimNet->forward(stream, specPrimInput, &specPrimOutput, use_inference_params, prepare_input_gradients);

    // Specular network of specular branch.
    GPUMatrix<float> clsInput(mpClsInput->data(), mClsInputDim, clsBatchSize);
    GPUMatrix<float> clsOutput(mpClsOutput->data(), mOutputDim, clsBatchSize);
    GPUMatrix<float> mergeInput(mpMergeInput->data(), mMergeInputDim, specBatchSize);
    GPUMatrix<float> mergeOutput(output->data() + diffBatchSize * mOutputDim, mOutputDim, specBatchSize);

    {
        HashGridInterp::launchForward(
            stream,
            mpSpecGrids,
            clsInput.data(),
            clsBatchSize,
            mClsInputDim,
            nLevels,
            nFeaturesPerLevel,
            log2HashMapSize,
            baseResolution,
            perLevelScale,
            interpRatio
        );
    }

    ctx->specCtx = mpSpecNet->forward(stream, clsInput, &clsOutput, use_inference_params, prepare_input_gradients);

    {
        const uint32_t count = specBatchSize;
        const uint32_t blockSize = 256;
        const uint32_t gridSize = (count + blockSize - 1) / blockSize;
        constructMergeInputKernel<<<gridSize, blockSize, 0, stream>>>(
            specPrimInput.data(),
            specPrimOutput.data(),
            clsInput.data(),
            clsOutput.data(),
            mergeInput.data(),
            count,
            numClusters,
            nFeaturesPerLevel,
            mPrimInputDim,
            mClsInputDim,
            mMergeInputDim,
            mOutputDim
        );
    }

    ctx->mergeCtx = mpMergeNet->forward(stream, mergeInput, &mergeOutput, use_inference_params, prepare_input_gradients);

    return ctx;
}

template<typename T>
void NeuralConeModel<T>::backward_impl(
    cudaStream_t stream,
    const Context& ctx,
    const GPUMatrixDynamic<T>& input,
    const GPUMatrixDynamic<T>& output,
    const GPUMatrixDynamic<T>& dL_doutput,
    GPUMatrixDynamic<T>* dL_dinput,
    bool use_inference_params,
    GradientMode param_gradients_mode
) {
    static_assert(std::is_same<T, float>::value, "NeuralConeModel currently supports float inference only.");

    const uint32_t batchSize = input.n();
    CHECK_THROW(batchSize <= mMaxBatchSize);
    const uint32_t diffBatchSize = mpDiffPrimInput->n();
    const uint32_t specBatchSize = mpSpecPrimInput->n();
    const uint32_t clsBatchSize = specBatchSize * numClusters;

    const NeuralConeModelContext<T>& ncCtx = static_cast<const NeuralConeModelContext<T>&>(ctx);
    if (param_gradients_mode == GradientMode::Overwrite) {
        CUDA_CHECK_THROW(cudaMemsetAsync(mpSpecGridsGradient, 0, sizeof(T) * mGridSize, stream));
    }

    if (specBatchSize > 0) {
        // Primary network of specular branch.
        GPUMatrix<float> specPrimInput(mpSpecPrimInput->data(), mPrimInputDim, specBatchSize);
        GPUMatrix<float> specPrimOutput(mpSpecPrimOutput->data(), mOutputDim, specBatchSize);

        GPUMatrix<float> dL_dspecPrimOutput(mpdLdSpecPrimOutput->data(), mOutputDim, specBatchSize);
        // GPUMatrix<float> dL_dinputSpec(dL_dinput->data() + diffBatchSize * mOutputDim, mPrimInputDim, specBatchSize);

        // Specular network of specular branch.
        GPUMatrix<float> clsInput(mpClsInput->data(), mClsInputDim, clsBatchSize);
        GPUMatrix<float> clsOutput(mpClsOutput->data(), mOutputDim, clsBatchSize);
        GPUMatrix<float> mergeInput(mpMergeInput->data(), mMergeInputDim, specBatchSize);
        GPUMatrix<float> mergeOutput(output.data() + diffBatchSize * mOutputDim, mOutputDim, specBatchSize);

        GPUMatrix<float> dL_dmergeOutput(dL_doutput.data() + diffBatchSize * mOutputDim, mOutputDim, specBatchSize);
        GPUMatrix<float> dL_dmergeInput(mpdLdMergeInput->data(), mMergeInputDim, specBatchSize);
        GPUMatrix<float> dL_dclsOutput(mpdLdClsOutput->data(), mOutputDim, clsBatchSize);
        GPUMatrix<float> dL_dclsInput(mpdLdClsInput->data(), mClsInputDim, clsBatchSize);

        mpMergeNet->backward(stream, *ncCtx.mergeCtx, mergeInput, mergeOutput, dL_dmergeOutput, &dL_dmergeInput, use_inference_params, param_gradients_mode);

        {
            const uint32_t count = specBatchSize;
            const uint32_t blockSize = 256;
            const uint32_t gridSize = (count + blockSize - 1) / blockSize;
            constructdLdClsOutputKernel<<<gridSize, blockSize, 0, stream>>>(
                // specPrimInput.data(),
                // specPrimOutput.data(),
                clsInput.data(),
                // clsOutput.data(),
                // mergeInput.data(),
                dL_dmergeInput.data(),
                dL_dclsOutput.data(),
                dL_dspecPrimOutput.data(),
                count,
                numClusters,
                nFeaturesPerLevel,
                // mPrimInputDim,
                mClsInputDim,
                mMergeInputDim,
                mOutputDim
            );
        }

        mpSpecNet->backward(stream, *ncCtx.specCtx, clsInput, clsOutput, dL_dclsOutput, &dL_dclsInput, use_inference_params, param_gradients_mode);

        HashGridInterp::launchBackward(
            stream,
            mpSpecGrids,
            clsInput.data(),
            dL_dclsInput.data(),
            mpSpecGridsGradient,
            clsBatchSize,
            mClsInputDim,
            nLevels,
            nFeaturesPerLevel,
            log2HashMapSize,
            baseResolution,
            perLevelScale,
            interpRatio
        );

        mpPrimNet->backward(stream, *ncCtx.specPrimCtx, specPrimInput, specPrimOutput, dL_dspecPrimOutput, nullptr/* &dL_dinputSpec */, use_inference_params, param_gradients_mode);

        // Clamp gradients to prevent training instability.
        {
            const uint32_t count = n_params();
            const uint32_t blockSize = 256;
            const uint32_t gridSize = (count + blockSize - 1) / blockSize;
            clampGradients<<<gridSize, blockSize, 0, stream>>>(this->gradients(), count, 1.0f);
        }
    }
    
    if (diffBatchSize > 0) {
        // Primary network of diffuse branch.
        GPUMatrix<float> diffPrimInput(mpDiffPrimInput->data(), mPrimInputDim, diffBatchSize);
        GPUMatrix<float> diffPrimOutput(output.data(), mOutputDim, diffBatchSize);

        GPUMatrix<float> dL_ddiffPrimOutput(dL_doutput.data(), mOutputDim, diffBatchSize);
        // GPUMatrix<float> dL_dinputDiff(dL_dinput->data(), mPrimInputDim, diffBatchSize);

        mpPrimNet->backward(stream, *ncCtx.diffPrimCtx, diffPrimInput, diffPrimOutput, dL_ddiffPrimOutput, nullptr/* &dL_dinputDiff */, use_inference_params, param_gradients_mode);
    }
}


template<typename T>
uint32_t NeuralConeModel<T>::input_width() const {
    return mPrimInputDim;
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
    uint32_t primNetOffset = 0;
    uint32_t specGridsOffset = primNetOffset + mpPrimNet->n_params();
    uint32_t specNetOffset = specGridsOffset + mGridSize;
    uint32_t mergeNetOffset = specNetOffset + mpSpecNet->n_params();

    mpPrimNet->set_params(params + primNetOffset, inference_params + primNetOffset, gradients + primNetOffset);
    mpSpecGrids = params + specGridsOffset;
    mpSpecGridsInference = inference_params + specGridsOffset;
    mpSpecGridsGradient = gradients + specGridsOffset;
    mpSpecNet->set_params(params + specNetOffset, inference_params + specNetOffset, gradients + specNetOffset);
    mpMergeNet->set_params(params + mergeNetOffset, inference_params + mergeNetOffset, gradients + mergeNetOffset);
}

template<typename T>
void NeuralConeModel<T>::initialize_params(pcg32& rnd, float* params_full_precision, float scale) {
    mpPrimNet->initialize_params(rnd, params_full_precision, scale);
    params_full_precision += mpPrimNet->n_params();
    mpSpecGrids = params_full_precision;
    params_full_precision += mGridSize;
    mpSpecNet->initialize_params(rnd, params_full_precision, scale);
    params_full_precision += mpSpecNet->n_params();
    mpMergeNet->initialize_params(rnd, params_full_precision, scale);
    params_full_precision += mpMergeNet->n_params();
}

template<typename T>
size_t NeuralConeModel<T>::n_params() const {
    return mpPrimNet->n_params() + mGridSize + mpSpecNet->n_params() + mpMergeNet->n_params();
}

template<typename T>
std::vector<std::pair<uint32_t, uint32_t>> NeuralConeModel<T>::layer_sizes() const {
    return {};
}

template class NeuralConeModel<float>;


}
