#include "Model.h"
#include "NeuralModel.h"
#include "NeuralConeModel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "tiny-cuda-nn/config.h"
#include "tiny-cuda-nn/loss.h"
#include "tiny-cuda-nn/optimizer.h"
#include "tiny-cuda-nn/trainer.h"


// Model code

namespace {
    using Trainer = tcnn::Trainer<float, precision_t, precision_t>;

    void syncTrainerParams(const Trainer& src, Trainer& dst) {
        dst.set_params(src.params_inference(), src.n_params(), true);
    }
}

NRModel::NRModel() {
    CUDA_CHECK_THROW(cudaStreamCreate(&mTrainStream));
    CUDA_CHECK_THROW(cudaStreamCreate(&mInferenceStream));

    createOnlineModels();

    mpDiffNet = mpDiffOnlineNet.get();
    mpSpecNet = mpSpecOnlineNet.get();
    mpDiffTrainer = mpDiffOnlineTrainer.get();
    mpSpecTrainer = mpSpecOnlineTrainer.get();
    mOnline = true;
}

void NRModel::createOnlineModels()
{
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

    mpDiffOnlineNet = std::make_shared<tcnn::NeuralModel<precision_t>>(gridConfig);
    {
        auto onlineOptimizer = std::shared_ptr<tcnn::Optimizer<precision_t>>(tcnn::create_optimizer<precision_t>(mOnlineOptConfig));
        auto onlineLoss = std::shared_ptr<tcnn::Loss<precision_t>>(tcnn::create_loss<precision_t>(mLossConfig));
        mpDiffOnlineTrainer = std::make_unique<Trainer>(mpDiffOnlineNet, onlineOptimizer, onlineLoss);
    }

    mpSpecOnlineNet = std::make_shared<tcnn::NeuralConeModel<precision_t>>(primGridConfig, clsGridConfig);
    {
        auto onlineOptimizer = std::shared_ptr<tcnn::Optimizer<precision_t>>(tcnn::create_optimizer<precision_t>(mOnlineOptConfig));
        auto onlineLoss = std::shared_ptr<tcnn::Loss<precision_t>>(tcnn::create_loss<precision_t>(mLossConfig));
        mpSpecOnlineTrainer = std::make_unique<Trainer>(mpSpecOnlineNet, onlineOptimizer, onlineLoss);
    }
}

void NRModel::createOfflineModelsIfNeeded()
{
    if (mpDiffOfflineNet && mpSpecOfflineNet && mpDiffOfflineTrainer && mpSpecOfflineTrainer) {
        return;
    }

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

    mpDiffOfflineNet = std::make_shared<tcnn::NeuralModel<precision_t>>(gridConfig);
    {
        auto offlineOptimizer = std::shared_ptr<tcnn::Optimizer<precision_t>>(tcnn::create_optimizer<precision_t>(mOfflineOptConfig));
        auto offlineLoss = std::shared_ptr<tcnn::Loss<precision_t>>(tcnn::create_loss<precision_t>(mLossConfig));
        mpDiffOfflineTrainer = std::make_unique<Trainer>(mpDiffOfflineNet, offlineOptimizer, offlineLoss);
    }

    mpSpecOfflineNet = std::make_shared<tcnn::NeuralConeModel<precision_t>>(primGridConfig, clsGridConfig);
    {
        auto offlineOptimizer = std::shared_ptr<tcnn::Optimizer<precision_t>>(tcnn::create_optimizer<precision_t>(mOfflineOptConfig));
        auto offlineLoss = std::shared_ptr<tcnn::Loss<precision_t>>(tcnn::create_loss<precision_t>(mLossConfig));
        mpSpecOfflineTrainer = std::make_unique<Trainer>(mpSpecOfflineNet, offlineOptimizer, offlineLoss);
    }
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

    if (mTrainStream) {
        CUDA_CHECK_THROW(cudaStreamDestroy(mTrainStream));
    }
    if (mInferenceStream) {
        CUDA_CHECK_THROW(cudaStreamDestroy(mInferenceStream));
    }
}

void NRModel::saveState(const std::string& path) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(mTrainStream));

    tcnn::json ckpt;
    ckpt["format"] = "nr_tcnn_ckpt_v1";
    ckpt["diff"] = mpDiffTrainer->serialize(true);
    ckpt["spec"] = mpSpecTrainer->serialize(true);

    std::ofstream ofs(path, std::ios::binary);
    const std::vector<std::uint8_t> cbor = tcnn::json::to_cbor(ckpt);
    ofs.write(reinterpret_cast<const char*>(cbor.data()), static_cast<std::streamsize>(cbor.size()));
}

void NRModel::loadState(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return;

    const std::vector<std::uint8_t> cbor((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    tcnn::json ckpt = tcnn::json::from_cbor(cbor);
    mpDiffTrainer->deserialize(ckpt.at("diff"));
    mpSpecTrainer->deserialize(ckpt.at("spec"));

    CUDA_CHECK_THROW(cudaStreamSynchronize(mInferenceStream));
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
        createOfflineModelsIfNeeded();
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
        mpDiffNet->inference(mInferenceStream, diffInput, diffOutput);
    }
    if (specBatchSize > 0) {
        mpSpecNet->inference(mInferenceStream, specInput, specOutput);
    }
}

void NRModel::train(ModelIOPtrs diffPtrs, ModelIOPtrs specPtrs) {
    const uint32_t diffBatchSize = padUp(diffPtrs.size, 256);
    const uint32_t specBatchSize = padUp(specPtrs.size, 256);

    tcnn::GPUMatrix<float> diffInput(diffPtrs.inputPtr, mDiffInputDim, diffBatchSize);
    tcnn::GPUMatrix<float> specInput(specPtrs.inputPtr, mSpecInputDim, specBatchSize);
    tcnn::GPUMatrix<float> diffOutput(diffPtrs.outputPtr, mOutputDim, diffBatchSize);
    tcnn::GPUMatrix<float> specOutput(specPtrs.outputPtr, mOutputDim, specBatchSize);

    if (diffBatchSize > 0) {
        mpDiffTrainer->training_step(mTrainStream, diffInput, diffOutput, nullptr, true, nullptr);
    }
    if (specBatchSize > 0) {
        mpSpecTrainer->training_step(mTrainStream, specInput, specOutput, nullptr, true, nullptr);
    }
}
