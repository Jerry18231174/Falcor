#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "Model.h"
#include "tiny-cuda-nn/config.h"

namespace tcnn {

template <typename T>
struct NeuralConeModelContext : public Context {
    uint32_t diffBatchSize = 0;
    uint32_t specBatchSize = 0;
    uint32_t clsBatchSize = 0;

    std::unique_ptr<Context> diffPrimCtx;
    std::unique_ptr<Context> specPrimCtx;
    std::unique_ptr<Context> specCtx;
    std::unique_ptr<Context> mergeCtx;
};

template <typename T>
class NeuralConeModel : public DifferentiableObject<T, T, T> {

public:
    NeuralConeModel();
    ~NeuralConeModel() = default;

    void setIOPtrs(
        GPUMatrix<T>& diffPrimInput,
        GPUMatrix<T>& specPrimInput,
        GPUMatrix<T>& clsInput
    ) {
        this->mpDiffPrimInput = &diffPrimInput;
        this->mpSpecPrimInput = &specPrimInput;
        this->mpClsInput = &clsInput;
    }

    void inference_mixed_precision_impl(
        cudaStream_t stream,
        const GPUMatrixDynamic<T>& input,
        GPUMatrixDynamic<T>& output,
        bool use_inference_params = true
    ) override;

    std::unique_ptr<Context> forward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<T>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override;

    void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override;

    uint32_t input_width() const override;
	uint32_t padded_output_width() const override;
	uint32_t output_width() const override;
	uint32_t required_input_alignment() const override;
    json hyperparams() const override;
    void set_params_impl(T* params, T* inference_params, T* gradients) override;
    void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override;
    size_t n_params() const override;
    std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override;
    
private:
    std::shared_ptr<NetworkWithInputEncoding<T>> mpPrimNet;

    T* mpSpecGrids;
    T* mpSpecGridsInference;
    T* mpSpecGridsGradient;
    uint32_t mGridSize = 0;
    std::shared_ptr<Network<T, T>> mpSpecNet;
    std::shared_ptr<Network<T, T>> mpMergeNet;

    GPUMatrix<T>* mpDiffPrimInput;
    GPUMatrix<T>* mpSpecPrimInput;
    GPUMatrix<T>* mpClsInput;
    std::unique_ptr<GPUMemory<float>> mpDiffPrimOutput;
    std::unique_ptr<GPUMemory<float>> mpSpecPrimOutput;
    std::unique_ptr<GPUMemory<float>> mpdLdSpecPrimOutput;
    std::unique_ptr<GPUMemory<float>> mpdLdClsInput;
    std::unique_ptr<GPUMemory<float>> mpClsOutput;
    std::unique_ptr<GPUMemory<float>> mpdLdClsOutput;
    std::unique_ptr<GPUMemory<float>> mpMergeInput;
    std::unique_ptr<GPUMemory<float>> mpdLdMergeInput;

    // Hyper parameters
    uint32_t nLevels = 8;
    uint32_t nFeaturesPerLevel = 8;
    uint32_t log2HashMapSize = 19;
    uint32_t baseResolution = 4;
    float perLevelScale = 2.0f;
    uint32_t numClusters = 4;
    float interpRatio = 0.5f;

    uint32_t mMaxBatchSize = 1u << 21;
    uint32_t mPrimInputDim = 3 + 3 * 4 + 1;
    uint32_t mClsInputDim = nFeaturesPerLevel + 3 * 2 + 1 + 1;
    uint32_t mMergeInputDim = 4 + 4 + 3 + 1 + 3;
    uint32_t mOutputDim = 16;
};

}
