#include "Model.h"
#include "NeuralConeModel.h"
#include <cuda_runtime.h>
#include <iostream>
#include "tiny-cuda-nn/config.h"
#include "tiny-cuda-nn/loss.h"
#include "tiny-cuda-nn/optimizer.h"
#include "tiny-cuda-nn/trainer.h"


__constant__ float3 cBBox[2];
__constant__ float cMaxScale;

// Device code
__global__ void copyPrimInputKernel(ModelIOPtrs ioPtrs, float* modelInput, const uint32_t size) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float* entry = modelInput + i * 16;

    // Position for Encoding & Network
    float3 pos = make_float3(ioPtrs.posPtr[i * 4 + 0], ioPtrs.posPtr[i * 4 + 1], ioPtrs.posPtr[i * 4 + 2]);
    entry[0] = (pos.x - cBBox[0].x) / (cBBox[1].x - cBBox[0].x);
    entry[1] = (pos.y - cBBox[0].y) / (cBBox[1].y - cBBox[0].y);
    entry[2] = (pos.z - cBBox[0].z) / (cBBox[1].z - cBBox[0].z);
    entry[3] = pos.x;
    entry[4] = pos.y;
    entry[5] = pos.z;

    // Direction
    entry[6] = ioPtrs.dirPtr[i * 4 + 0];
    entry[7] = ioPtrs.dirPtr[i * 4 + 1];
    entry[8] = ioPtrs.dirPtr[i * 4 + 2];

    // Normal
    entry[9] = ioPtrs.normalPtr[i * 4 + 0];
    entry[10] = ioPtrs.normalPtr[i * 4 + 1];
    entry[11] = ioPtrs.normalPtr[i * 4 + 2];

    // Albedo
    entry[12] = ioPtrs.albedoPtr[i * 4 + 0];
    entry[13] = ioPtrs.albedoPtr[i * 4 + 1];
    entry[14] = ioPtrs.albedoPtr[i * 4 + 2];

    // Roughness
    entry[15] = ioPtrs.roughnessPtr[i];
}

template<uint32_t N_FEATURES>
__global__ void copyClsInputKernel(ModelIOPtrs ioPtrs, float* modelInput, const uint32_t numClusters, const uint32_t size) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float* entry = modelInput + i * numClusters * (N_FEATURES + 8);

    for (int j = 0; j < numClusters; j++) {
        entry[j * (N_FEATURES + 8) + N_FEATURES + 0] = (ioPtrs.clsPosPtr[(i * numClusters + j) * 4 + 0] - cBBox[0].x) / (cBBox[1].x - cBBox[0].x);
        entry[j * (N_FEATURES + 8) + N_FEATURES + 1] = (ioPtrs.clsPosPtr[(i * numClusters + j) * 4 + 1] - cBBox[0].y) / (cBBox[1].y - cBBox[0].y);
        entry[j * (N_FEATURES + 8) + N_FEATURES + 2] = (ioPtrs.clsPosPtr[(i * numClusters + j) * 4 + 2] - cBBox[0].z) / (cBBox[1].z - cBBox[0].z);
        entry[j * (N_FEATURES + 8) + N_FEATURES + 3] = ioPtrs.clsDirPtr[(i * numClusters + j) * 4 + 0];
        entry[j * (N_FEATURES + 8) + N_FEATURES + 4] = ioPtrs.clsDirPtr[(i * numClusters + j) * 4 + 1];
        entry[j * (N_FEATURES + 8) + N_FEATURES + 5] = ioPtrs.clsDirPtr[(i * numClusters + j) * 4 + 2];
        entry[j * (N_FEATURES + 8) + N_FEATURES + 6] = ioPtrs.clsScalePtr[i * numClusters + j] / cMaxScale;
        entry[j * (N_FEATURES + 8) + N_FEATURES + 7] = ioPtrs.clsWeightPtr[i * numClusters + j];
    }
}

template<uint32_t DIM>
__global__ void copyOutputKernel(ModelIOPtrs ioPtrs, const float* modelOutput, const uint32_t size) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    const float* entry = modelOutput + i * DIM;

    #pragma unroll
    for (uint32_t j = 0; j < DIM; j++) {
        ioPtrs.outputPtr[i * DIM + j] = entry[j];
    }
}

template<uint32_t DIM>
__global__ void copyOutputBackwardKernel(ModelIOPtrs ioPtrs, float* modelOutput, const uint32_t size) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float* entry = modelOutput + i * DIM;

    #pragma unroll
    for (uint32_t j = 0; j < DIM; j++) {
        entry[j] = ioPtrs.outputPtr[i * DIM + j];
    }
}

// Host code
void launchCopyPrimInput(ModelIOPtrs ioPtrs, float* modelInput, const uint32_t size, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((size + 255) / 256);

    copyPrimInputKernel<<<grid, block, 0, stream>>>(ioPtrs, modelInput, size);
}

void launchCopyClsInput(
    ModelIOPtrs ioPtrs,
    float* modelInput,
    const uint32_t numClusters,
    const uint32_t size,
    const uint32_t nFeaturesPerLevel,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((size + 255) / 256);

    if (nFeaturesPerLevel == 4) {
        copyClsInputKernel<4><<<grid, block, 0, stream>>>(ioPtrs, modelInput, numClusters, size);
    } else if (nFeaturesPerLevel == 8) {
        copyClsInputKernel<8><<<grid, block, 0, stream>>>(ioPtrs, modelInput, numClusters, size);
    } else {
        assert(false && "Unsupported number of features per level");
    }
}

void launchCopyOutput(ModelIOPtrs ioPtrs, const float* modelOutput, const uint32_t size, const uint32_t outputDim, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((size + 255) / 256);

    if (outputDim == 4) {
        copyOutputKernel<4><<<grid, block, 0, stream>>>(ioPtrs, modelOutput, size);
    } else if (outputDim == 16) {
        copyOutputKernel<16><<<grid, block, 0, stream>>>(ioPtrs, modelOutput, size);
    } else {
        assert(false && "Unsupported output dimension");
    }
}

void launchCopyOutputBackward(
    ModelIOPtrs ioPtrs,
    float* modelOutput,
    const uint32_t size,
    const uint32_t outputDim,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((size + 255) / 256);

    if (outputDim == 4) {
        copyOutputBackwardKernel<4><<<grid, block, 0, stream>>>(ioPtrs, modelOutput, size);
    } else if (outputDim == 16) {
        copyOutputBackwardKernel<16><<<grid, block, 0, stream>>>(ioPtrs, modelOutput, size);
    } else {
        assert(false && "Unsupported output dimension");
    }
}


// Model code

uint32_t padUp(uint32_t x, uint32_t align) { return (x + align - 1) & ~(align - 1); }

NRModel::NRModel(const std::vector<float> bbox) {
    CUDA_CHECK_THROW(cudaStreamCreate(&mStream));

    // Set bbox constants
    const float maxScale = max(bbox[3] - bbox[0], max(bbox[4] - bbox[1], bbox[5] - bbox[2]));
    cudaMemcpyToSymbol(cBBox, bbox.data(), 6 * sizeof(float));
    cudaMemcpyToSymbol(cMaxScale, &maxScale, sizeof(float));

    mpNet = std::make_shared<tcnn::NeuralConeModel<float>>();
    {
        auto optimizer = std::shared_ptr<tcnn::Optimizer<float>>(tcnn::create_optimizer<float>({
            {"otype", "Adam"},
            {"learning_rate", 1e-3f}
        }));
        auto loss = std::shared_ptr<tcnn::Loss<float>>(tcnn::create_loss<float>({{"otype", "RelativeL2Luminance"}}));
        mpTrainer = std::make_unique<tcnn::Trainer<float, float, float>>(mpNet, optimizer, loss);
    }
    mpDiffPrimInput = std::make_shared<tcnn::GPUMemory<float>>(mPrimInputDim * padUp(pixelCount, 256));
    mpSpecPrimInput = std::make_shared<tcnn::GPUMemory<float>>(mPrimInputDim * padUp(pixelCount, 256));
    mpClsInput = std::make_shared<tcnn::GPUMemory<float>>(mClsInputDim * numClusters * padUp(pixelCount, 256));
    mpOutput = std::make_shared<tcnn::GPUMemory<float>>(mOutputDim * padUp(pixelCount, 256));
}

NRModel::~NRModel() {
    mpTrainer.reset();
    mpNet.reset();
    mpDiffPrimInput.reset();
    mpSpecPrimInput.reset();
    mpClsInput.reset();
    mpOutput.reset();
    
    if (mStream) {
        CUDA_CHECK_THROW(cudaStreamDestroy(mStream));
    }
}

void NRModel::inference(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs, const uint32_t diffSize, const uint32_t specSize) {
    const uint32_t diffBatchSize = padUp(diffSize, 256);
    const uint32_t specBatchSize = padUp(specSize, 256);

    CUDA_CHECK_THROW(cudaStreamSynchronize(mStream));
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    // Copy primary G-Buffers
    if (diffSize > 0) launchCopyPrimInput(diffPtrs, mpDiffPrimInput->data(), diffSize, mStream);
    if (specSize > 0) {
        launchCopyPrimInput(specPtrs, mpSpecPrimInput->data(), specSize, mStream);
        launchCopyClsInput(specPtrs, mpClsInput->data(), numClusters, specSize, nFeaturesPerLevel, mStream);
    }

    CUDA_CHECK_THROW(cudaStreamSynchronize(mStream));
    std::chrono::steady_clock::time_point copyInputEnd = std::chrono::steady_clock::now();

    // modelInput is dummy input, we don't use its gradient.
    tcnn::GPUMatrix<float> modelInput(mpDiffPrimInput->data(), mPrimInputDim, diffBatchSize + specBatchSize);
    // modelOutput holds real output, need to copy.
    tcnn::GPUMatrix<float> modelOutput(mpOutput->data(), mOutputDim, diffBatchSize + specBatchSize);
    tcnn::GPUMatrix<float> diffPrimInput(mpDiffPrimInput->data(), mPrimInputDim, diffBatchSize);
    tcnn::GPUMatrix<float> specPrimInput(mpSpecPrimInput->data(), mPrimInputDim, specBatchSize);
    tcnn::GPUMatrix<float> clsInput(mpClsInput->data(), mClsInputDim, numClusters * specBatchSize);

    mpNet->setIOPtrs(diffPrimInput, specPrimInput, clsInput);
    mpNet->inference(mStream, modelInput, modelOutput);

    CUDA_CHECK_THROW(cudaStreamSynchronize(mStream));
    std::chrono::steady_clock::time_point inferenceEnd = std::chrono::steady_clock::now();

    // Copy output to respective outputPtrs.
    if (diffSize > 0) launchCopyOutput(diffPtrs, modelOutput.data(), diffSize, mOutputDim, mStream);
    if (specSize > 0) launchCopyOutput(specPtrs, modelOutput.data() + diffBatchSize * mOutputDim, specSize, mOutputDim, mStream);

    CUDA_CHECK_THROW(cudaStreamSynchronize(mStream));
    std::chrono::steady_clock::time_point copyOutputEnd = std::chrono::steady_clock::now();

    // Display in milliseconds
    std::chrono::duration<double, std::milli> copyInputTime = copyInputEnd - start;
    std::chrono::duration<double, std::milli> inferenceTime = inferenceEnd - copyInputEnd;
    std::chrono::duration<double, std::milli> copyOutputTime = copyOutputEnd - inferenceEnd;
    // std::cout << "Copy Input Time: " << copyInputTime.count() << " ms, Inference Time: " << inferenceTime.count() << " ms, Copy Output Time: " << copyOutputTime.count() << " ms" << std::endl;
}

void NRModel::train(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs, const uint32_t diffSize, const uint32_t specSize) {
    const uint32_t diffBatchSize = padUp(diffSize, 256);
    const uint32_t specBatchSize = padUp(specSize, 256);

    // Copy primary G-Buffers
    if (diffSize > 0) launchCopyPrimInput(diffPtrs, mpDiffPrimInput->data(), diffSize, mStream);
    if (specSize > 0) {
        launchCopyPrimInput(specPtrs, mpSpecPrimInput->data(), specSize, mStream);
        launchCopyClsInput(specPtrs, mpClsInput->data(), numClusters, specSize, nFeaturesPerLevel, mStream);
    }

    // modelInput is dummy input, we don't use its gradient.
    tcnn::GPUMatrix<float> modelInput(mpDiffPrimInput->data(), mPrimInputDim, diffBatchSize + specBatchSize);
    // modelOutput holds real output, need to copy.
    tcnn::GPUMatrix<float> modelOutput(mpOutput->data(), mOutputDim, diffBatchSize + specBatchSize);
    tcnn::GPUMatrix<float> diffPrimInput(mpDiffPrimInput->data(), mPrimInputDim, diffBatchSize);
    tcnn::GPUMatrix<float> specPrimInput(mpSpecPrimInput->data(), mPrimInputDim, specBatchSize);
    tcnn::GPUMatrix<float> clsInput(mpClsInput->data(), mClsInputDim, numClusters * specBatchSize);

    // Copy (target) output to respective outputPtrs.
    if (diffSize > 0) launchCopyOutputBackward(diffPtrs, modelOutput.data(), diffSize, mOutputDim, mStream);
    if (specSize > 0) launchCopyOutputBackward(specPtrs, modelOutput.data() + diffBatchSize * mOutputDim, specSize, mOutputDim, mStream);

    mpNet->setIOPtrs(diffPrimInput, specPrimInput, clsInput);
    mpTrainer->training_step(mStream, modelInput, modelOutput);
}
