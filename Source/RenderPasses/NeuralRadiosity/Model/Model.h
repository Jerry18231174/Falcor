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

    // Custom models
    template<typename T>
    class NeuralModel;

    template<typename T>
    class NeuralConeModel;
}

struct ModelIOPtrs
{
    float* inputPtr;
    float* outputPtr;

    uint32_t size;
};

inline constexpr uint32_t padUp(uint32_t x, uint32_t align) { return (x + align - 1) & ~(align - 1); }


class NRModel
{
public:
    NRModel();
    ~NRModel();

    void loadState();
    void saveState();

    void inference(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs);
    void train(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs);

    cudaStream_t stream() const { return mStream; }

private:
    std::shared_ptr<tcnn::NeuralModel<float>> mpDiffNet;
    std::shared_ptr<tcnn::NeuralConeModel<float>> mpSpecNet;
    std::unique_ptr<tcnn::Trainer<float, float, float>> mpDiffTrainer;
    std::unique_ptr<tcnn::Trainer<float, float, float>> mpSpecTrainer;
    cudaStream_t mStream;

    std::shared_ptr<tcnn::GPUMemory<float>> mpdLdDiffInput;
    std::shared_ptr<tcnn::GPUMemory<float>> mpdLdSpecInput;

    uint32_t pixelCount = 1u << 21;
    uint32_t numClusters = 4;

    // Hyper parameters
    uint32_t nLevels = 4;
    uint32_t nFeaturesPerLevel = 8;
    uint32_t log2HashMapSize = 19;
    uint32_t baseResolution = 32;
    float perLevelScale = 2.0f;

    uint32_t mDiffInputDim = padUp(nFeaturesPerLevel * nLevels + 3 * 4 + 1, 16);
    uint32_t mSpecInputDim = padUp(nFeaturesPerLevel * (nLevels + numClusters) + 3 * 4 + 1 + numClusters * (3 + 1 + 1), 16);
    uint32_t mOutputDim = 16;
};
