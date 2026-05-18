#include "Model.h"
#include "NeuralModel.h"
#include "NeuralConeModel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>
#include "tiny-cuda-nn/config.h"
#include "tiny-cuda-nn/loss.h"
#include "tiny-cuda-nn/optimizer.h"
#include "tiny-cuda-nn/trainer.h"


// Model code

namespace {
    using Trainer = tcnn::Trainer<float, float, float>;

    void syncTrainerParams(const Trainer& src, Trainer& dst) {
        dst.set_params(src.params_inference(), src.n_params(), true);
    }
}

NRModel::NRModel() {
    CUDA_CHECK_THROW(cudaStreamCreate(&mStream));

    HashGrid::Config gridConfig{nLevels, nFeaturesPerLevel, log2HashMapSize, baseResolution, perLevelScale};
    HashGrid::Config primGridConfig{nLevels, nFeaturesPerLevel, log2HashMapSize, baseResolution, perLevelScale};
    HashGridInterp::Config clsGridConfig{
        numClusters,
        nInterpLevels,
        nInterpFeaturesPerLevel,
        log2InterpHashMapSize,
        baseInterpResolution,
        perLevelInterpScale,
        interpRatio
    };

    nlohmann::json offlineOptConfig = nlohmann::json({
        {"otype", "Adam"},
        {"learning_rate", 1e-2f}
    });
    nlohmann::json onlineOptConfig = nlohmann::json({
        {"otype", "EMA"},
        {"decay", 0.9f},
        {"nested", offlineOptConfig}
    });
    nlohmann::json lossConfig = nlohmann::json({
        {"otype", "RelativeL2Luminance"}
    });

    mpDiffOfflineNet = std::make_shared<tcnn::NeuralModel<float>>(gridConfig);
    mpDiffOnlineNet = std::make_shared<tcnn::NeuralModel<float>>(gridConfig);
    {
        auto offlineOptimizer = std::shared_ptr<tcnn::Optimizer<float>>(tcnn::create_optimizer<float>(offlineOptConfig));
        auto onlineOptimizer = std::shared_ptr<tcnn::Optimizer<float>>(tcnn::create_optimizer<float>(onlineOptConfig));
        auto offlineLoss = std::shared_ptr<tcnn::Loss<float>>(tcnn::create_loss<float>(lossConfig));
        auto onlineLoss = std::shared_ptr<tcnn::Loss<float>>(tcnn::create_loss<float>(lossConfig));
        mpDiffOfflineTrainer = std::make_unique<Trainer>(mpDiffOfflineNet, offlineOptimizer, offlineLoss);
        mpDiffOnlineTrainer = std::make_unique<Trainer>(mpDiffOnlineNet, onlineOptimizer, onlineLoss);
    }

    mpSpecOfflineNet = std::make_shared<tcnn::NeuralConeModel<float>>(primGridConfig, clsGridConfig);
    mpSpecOnlineNet = std::make_shared<tcnn::NeuralConeModel<float>>(primGridConfig, clsGridConfig);
    {
        auto offlineOptimizer = std::shared_ptr<tcnn::Optimizer<float>>(tcnn::create_optimizer<float>(offlineOptConfig));
        auto onlineOptimizer = std::shared_ptr<tcnn::Optimizer<float>>(tcnn::create_optimizer<float>(onlineOptConfig));
        auto offlineLoss = std::shared_ptr<tcnn::Loss<float>>(tcnn::create_loss<float>(lossConfig));
        auto onlineLoss = std::shared_ptr<tcnn::Loss<float>>(tcnn::create_loss<float>(lossConfig));
        mpSpecOfflineTrainer = std::make_unique<Trainer>(mpSpecOfflineNet, offlineOptimizer, offlineLoss);
        mpSpecOnlineTrainer = std::make_unique<Trainer>(mpSpecOnlineNet, onlineOptimizer, onlineLoss);
    }

    mpdLdDiffInput = std::make_shared<tcnn::GPUMemory<float>>(mDiffInputDim * padUp(pixelCount, 256));
    mpdLdSpecInput = std::make_shared<tcnn::GPUMemory<float>>(mSpecInputDim * padUp(pixelCount, 256));

    mpDiffNet = mpDiffOfflineNet.get();
    mpSpecNet = mpSpecOfflineNet.get();
    mpDiffTrainer = mpDiffOfflineTrainer.get();
    mpSpecTrainer = mpSpecOfflineTrainer.get();
    syncTrainerParams(*mpDiffOfflineTrainer, *mpDiffOnlineTrainer);
    syncTrainerParams(*mpSpecOfflineTrainer, *mpSpecOnlineTrainer);
}

NRModel::~NRModel() {
    mpDiffNet = nullptr;
    mpSpecNet = nullptr;
    mpDiffTrainer = nullptr;
    mpSpecTrainer = nullptr;
    mpDiffOfflineTrainer.reset();
    mpDiffOnlineTrainer.reset();
    mpDiffOfflineNet.reset();
    mpDiffOnlineNet.reset();
    mpSpecOfflineTrainer.reset();
    mpSpecOnlineTrainer.reset();
    mpSpecOfflineNet.reset();
    mpSpecOnlineNet.reset();

    if (mStream) {
        CUDA_CHECK_THROW(cudaStreamDestroy(mStream));
    }
}

void NRModel::saveState(const std::string& path) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(mStream));

    nlohmann::json ckpt;
    ckpt["format"] = "nr_tcnn_ckpt_v1";
    ckpt["diff"] = mpDiffTrainer->serialize(true);
    ckpt["spec"] = mpSpecTrainer->serialize(true);

    std::ofstream ofs(path, std::ios::binary);
    const std::vector<std::uint8_t> cbor = nlohmann::json::to_cbor(ckpt);
    ofs.write(reinterpret_cast<const char*>(cbor.data()), static_cast<std::streamsize>(cbor.size()));
}

void NRModel::loadState(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return;

    const std::vector<std::uint8_t> cbor((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    nlohmann::json ckpt = nlohmann::json::from_cbor(cbor);
    mpDiffTrainer->deserialize(ckpt.at("diff"));
    mpSpecTrainer->deserialize(ckpt.at("spec"));

    CUDA_CHECK_THROW(cudaStreamSynchronize(mStream));
}

void NRModel::setOnline(bool online)
{
    if (online == mOnline) {
        return;
    }

    if (online) {
        syncTrainerParams(*mpDiffTrainer, *mpDiffOnlineTrainer);
        syncTrainerParams(*mpSpecTrainer, *mpSpecOnlineTrainer);
        mpDiffNet = mpDiffOnlineNet.get();
        mpSpecNet = mpSpecOnlineNet.get();
        mpDiffTrainer = mpDiffOnlineTrainer.get();
        mpSpecTrainer = mpSpecOnlineTrainer.get();
    } else {
        syncTrainerParams(*mpDiffTrainer, *mpDiffOfflineTrainer);
        syncTrainerParams(*mpSpecTrainer, *mpSpecOfflineTrainer);
        mpDiffNet = mpDiffOfflineNet.get();
        mpSpecNet = mpSpecOfflineNet.get();
        mpDiffTrainer = mpDiffOfflineTrainer.get();
        mpSpecTrainer = mpSpecOfflineTrainer.get();
    }

    mOnline = online;
}

void NRModel::inference(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs) {
    const uint32_t diffBatchSize = padUp(diffPtrs.size, 256);
    const uint32_t specBatchSize = padUp(specPtrs.size, 256);

    tcnn::GPUMatrix<float> diffInput(diffPtrs.inputPtr, mDiffInputDim, diffBatchSize);
    tcnn::GPUMatrix<float> specInput(specPtrs.inputPtr, mSpecInputDim, specBatchSize);
    tcnn::GPUMatrix<float> diffOutput(diffPtrs.outputPtr, mOutputDim, diffBatchSize);
    tcnn::GPUMatrix<float> specOutput(specPtrs.outputPtr, mOutputDim, specBatchSize);

    if (diffBatchSize > 0) {
        mpDiffNet->inference(mStream, diffInput, diffOutput);
    }
    if (specBatchSize > 0) {
        mpSpecNet->inference(mStream, specInput, specOutput);
    }
}

void NRModel::train(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs) {
    const uint32_t diffBatchSize = padUp(diffPtrs.size, 256);
    const uint32_t specBatchSize = padUp(specPtrs.size, 256);

    tcnn::GPUMatrix<float> diffInput(diffPtrs.inputPtr, mDiffInputDim, diffBatchSize);
    tcnn::GPUMatrix<float> specInput(specPtrs.inputPtr, mSpecInputDim, specBatchSize);
    tcnn::GPUMatrix<float> diffOutput(diffPtrs.outputPtr, mOutputDim, diffBatchSize);
    tcnn::GPUMatrix<float> specOutput(specPtrs.outputPtr, mOutputDim, specBatchSize);

    tcnn::GPUMatrix<float> dLdDiffInput(mpdLdDiffInput->data(), mDiffInputDim, diffBatchSize);
    tcnn::GPUMatrix<float> dLdSpecInput(mpdLdSpecInput->data(), mSpecInputDim, specBatchSize);

    if (diffBatchSize > 0) {
        mpDiffTrainer->training_step(mStream, diffInput, diffOutput, nullptr, true, &dLdDiffInput);
    }
    if (specBatchSize > 0) {
        mpSpecTrainer->training_step(mStream, specInput, specOutput, nullptr, true, &dLdSpecInput);
    }
}
