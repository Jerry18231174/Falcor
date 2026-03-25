#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "Model.h"
#include "HashGrid.h"
#include "tiny-cuda-nn/config.h"

namespace tcnn {

template <typename T>
struct NeuralModelContext : public Context {
    uint32_t batchSize = 0;

    std::unique_ptr<Context> netCtx;
};

template <typename T>
class NeuralModel : public DifferentiableObject<T, T, T> {

public:
    NeuralModel();
    ~NeuralModel() = default;

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
    std::shared_ptr<Network<T, T>> mpNet;

    T* mpGrids;
    T* mpGridsInference;
    T* mpGridsGradient;
    uint32_t mGridSize = 0;

    HashGrid::Config mGridConfig = {4, 8, 19, 32, 2.0f};

    uint32_t encOffset = 0;
    uint32_t posOffset = encOffset + mGridConfig.nLevels * mGridConfig.nFeaturesPerLevel;
    uint32_t dirOffset = posOffset + 3;
    uint32_t normalOffset = dirOffset + 3;
    uint32_t albedoOffset = normalOffset + 3;
    uint32_t roughnessOffset = albedoOffset + 3;
    uint32_t totalDim = roughnessOffset + 1;

    uint32_t mInputDim = padUp(totalDim, 16);
    uint32_t mOutputDim = 16;
    uint32_t mMaxBatchSize = 1u << 21;
};

}
