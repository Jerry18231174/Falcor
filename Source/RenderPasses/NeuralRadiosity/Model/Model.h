#pragma once
#include <cstdint>
#include <memory>


namespace tcnn
{
    template <typename T>
    class GPUMemory;

    template <typename T>
    class Encoding;

    template <typename T, typename PARAMS_T>
    class Network;

    template <typename T>
    class NetworkWithInputEncoding;

    template <typename T, typename PARAMS_T, typename COMPUTE_T>
    class Trainer;

    // Custom neural cone model
    template<typename T>
    class NeuralConeModel;
}

struct ModelIOPtrs
{
    const float* posPtr;
    const float* dirPtr;
    const float* normalPtr;
    const float* albedoPtr;
    const float* roughnessPtr;

    const float* clsPosPtr;
    const float* clsDirPtr;
    const float* clsScalePtr;
    const float* clsWeightPtr;

    float* outputPtr;
};


class NRModel
{
public:
    NRModel(const std::vector<float> bbox);
    ~NRModel();

    void loadState();
    void saveState();

    void inference(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs, const uint32_t diffSize, const uint32_t specSize);
    void train(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs, const uint32_t diffSize, const uint32_t specSize);

    cudaStream_t stream() const { return mStream; }

private:
    std::shared_ptr<tcnn::NeuralConeModel<float>> mpNet;

    std::shared_ptr<tcnn::GPUMemory<float>> mpDiffPrimInput;
    std::shared_ptr<tcnn::GPUMemory<float>> mpSpecPrimInput;
    std::shared_ptr<tcnn::GPUMemory<float>> mpClsInput;
    std::shared_ptr<tcnn::GPUMemory<float>> mpOutput;

    std::unique_ptr<tcnn::Trainer<float, float, float>> mpTrainer;
    cudaStream_t mStream;

    uint32_t pixelCount = 1u << 21;
    uint32_t numClusters = 4;

    // Hyper parameters
    uint32_t nLevels = 4;
    uint32_t nFeaturesPerLevel = 8;
    uint32_t log2HashMapSize = 19;
    uint32_t baseResolution = 32;
    float perLevelScale = 2.0f;

    uint32_t mPrimInputDim = 3 + 3 * 4 + 1;
    uint32_t mClsInputDim = nFeaturesPerLevel + 3 * 2 + 1 + 1;
    uint32_t mOutputDim = 16;
};
