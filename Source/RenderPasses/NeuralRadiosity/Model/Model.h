#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include "tiny-cuda-nn/object.h"


#if defined(TCNN_HALF_PRECISION) && TCNN_HALF_PRECISION && defined(__CUDACC__)
#include <cuda_fp16.h>
#elif defined(TCNN_HALF_PRECISION) && TCNN_HALF_PRECISION
struct __half;
#endif

#if defined(TCNN_HALF_PRECISION) && TCNN_HALF_PRECISION
using precision_t = __half;
#else
using precision_t = float;
#endif


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

    void loadState(const std::string& path);
    void saveState(const std::string& path);

    void setOnline(bool online);

    void inference(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs);
    void train(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs);

    cudaStream_t stream() const { return mStream; }

private:
    void createOnlineModels();
    void createOfflineModelsIfNeeded();

    std::shared_ptr<tcnn::NeuralModel<precision_t>> mpDiffOfflineNet;
    std::shared_ptr<tcnn::NeuralModel<precision_t>> mpDiffOnlineNet;
    std::shared_ptr<tcnn::NeuralConeModel<precision_t>> mpSpecOfflineNet;
    std::shared_ptr<tcnn::NeuralConeModel<precision_t>> mpSpecOnlineNet;
    tcnn::NeuralModel<precision_t>* mpDiffNet = nullptr;
    tcnn::NeuralConeModel<precision_t>* mpSpecNet = nullptr;
    std::unique_ptr<tcnn::Trainer<float, precision_t, precision_t>> mpDiffOfflineTrainer;
    std::unique_ptr<tcnn::Trainer<float, precision_t, precision_t>> mpSpecOfflineTrainer;
    std::unique_ptr<tcnn::Trainer<float, precision_t, precision_t>> mpDiffOnlineTrainer;
    std::unique_ptr<tcnn::Trainer<float, precision_t, precision_t>> mpSpecOnlineTrainer;
    tcnn::Trainer<float, precision_t, precision_t>* mpDiffTrainer = nullptr;
    tcnn::Trainer<float, precision_t, precision_t>* mpSpecTrainer = nullptr;
    cudaStream_t mStream;
    bool mOnline = false;

    std::shared_ptr<tcnn::GPUMemory<float>> mpdLdDiffInput;
    std::shared_ptr<tcnn::GPUMemory<float>> mpdLdSpecInput;

    uint32_t pixelCount = 1u << 20;
    uint32_t numClusters = 4;

    // Hyper parameters
    uint32_t nLevels = 6;
    uint32_t nFeaturesPerLevel = 4;
    uint32_t log2HashMapSize = 19;
    uint32_t baseResolution = 32;
    float perLevelScale = 2.0f;

    uint32_t nInterpLevels = 8;
    uint32_t nInterpFeaturesPerLevel = 8;
    uint32_t log2InterpHashMapSize = 19;
    uint32_t baseInterpResolution = 4;
    float perLevelInterpScale = 2.0f;
    float interpRatio = 0.5f;

    const tcnn::json mOfflineOptConfig = {
        {"otype", "Adam"},
        {"learning_rate", 1e-2f}
    };
    const tcnn::json mOnlineOptConfig = {
        {"otype", "EMA"},
        {"decay", 0.9f},
        {"nested", mOfflineOptConfig}
    };
    const tcnn::json mLossConfig = {
        {"otype", "RelativeL2Luminance"}
    };

    uint32_t mDiffInputDim = padUp(nFeaturesPerLevel * nLevels + 3 * 4 + 1, 16);
    uint32_t mSpecInputDim = padUp(nFeaturesPerLevel * nLevels + nInterpFeaturesPerLevel * numClusters + 3 * 4 + 1 + numClusters * (3 + 1 + 1), 16);
    uint32_t mOutputDim = 16;
};
