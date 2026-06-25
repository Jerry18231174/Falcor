#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "Model.h"
#include "HashGrid.h"
#include "HashGridInterp.h"
#include "tiny-cuda-nn/config.h"

namespace tcnn {

template <typename T>
struct NeuralConeModelContext : public Context {
    uint32_t batchSize = 0;
    GPUMatrix<T> networkInput;
    std::unique_ptr<Context> netCtx;
};

template <typename T>
class NeuralConeModel : public DifferentiableObject<float, T, T> {

public:
    NeuralConeModel(HashGrid::Config primGridConfig, HashGridInterp::Config clsGridConfig);
    ~NeuralConeModel() = default;

    void inference_mixed_precision_impl(
        cudaStream_t stream,
        const GPUMatrixDynamic<float>& input,
        GPUMatrixDynamic<T>& output,
        bool use_inference_params = true
    ) override;

    std::unique_ptr<Context> forward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override;

    void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
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
    std::shared_ptr<Network<T, T>> mpNet;

    T* mpPrimGrids;
    T* mpPrimGridsInference;
    T* mpPrimGridsGradient;
    uint32_t mPrimGridSize = 0;

    T* mpClsGrids;
    T* mpClsGridsInference;
    T* mpClsGridsGradient;
    uint32_t mClsGridSize = 0;

    HashGrid::Config mPrimGridConfig;
    HashGridInterp::Config mClsGridConfig;

    uint32_t primEncOffset = 0;
    uint32_t clsEncOffset = primEncOffset + mPrimGridConfig.nLevels * mPrimGridConfig.nFeaturesPerLevel;
    uint32_t posOffset = clsEncOffset + mClsGridConfig.nClusters * mClsGridConfig.nFeaturesPerLevel;
    uint32_t dirOffset = posOffset + 3;
    uint32_t normalOffset = dirOffset + 3;
    uint32_t albedoOffset = normalOffset + 3;
    uint32_t roughnessOffset = albedoOffset + 3;
    uint32_t clsPosOffset = roughnessOffset + 1;
    uint32_t clsScaleOffset = clsPosOffset + mClsGridConfig.nClusters * 3;
    uint32_t clsWeightOffset = clsScaleOffset + mClsGridConfig.nClusters;
    uint32_t totalDim = clsWeightOffset + mClsGridConfig.nClusters;
    
    uint32_t mInputDim = padUp(totalDim, 16);
    uint32_t mOutputDim = 16;
    uint32_t mMaxBatchSize = 1u << 21;
};

}
