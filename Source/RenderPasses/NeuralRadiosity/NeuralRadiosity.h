/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Utils/CudaUtils.h"
#include "Utils/Algorithm/PrefixSum.h"
#include "Model/Model.h"

#include <filesystem>
#include <memory>

using namespace Falcor;

struct RayBatchBuffer
{
    uint32_t size = 0;
    uint32_t diffSize = 0;
    uint32_t specSize = 0;
    uint32_t numClusters = 0;

    /// G-buffer
    ref<Buffer> pos;
    ref<Buffer> dir;
    ref<Buffer> normal;
    ref<Buffer> albedo;
    ref<Buffer> roughness;
    ref<Buffer> vbuffer;
    ref<Buffer> color;
    /// Diffuse buffer
    ref<Buffer> diffPos;
    ref<Buffer> diffDir;
    ref<Buffer> diffNormal;
    ref<Buffer> diffAlbedo;
    ref<Buffer> diffRoughness;
    ref<Buffer> diffActive;         // Compaction Input
    ref<Buffer> diffIndex;          // Compaction Output
    ref<Buffer> diffColor;
    /// Specular buffer
    ref<Buffer> specPos;
    ref<Buffer> specDir;
    ref<Buffer> specNormal;
    ref<Buffer> specAlbedo;
    ref<Buffer> specRoughness;
    ref<Buffer> specVBuffer;
    ref<Buffer> specActive;       // Compaction Input
    ref<Buffer> specIndex;        // Compaction Output
    ref<Buffer> specColor;
    /// Cluster position, direction, scale, and weight
    ref<Buffer> clusterPos;
    ref<Buffer> clusterDir;
    ref<Buffer> clusterScale;
    ref<Buffer> clusterWeight;
    /// Corresponding pointers
    ModelIOPtrs diffPtrs;
    ModelIOPtrs specPtrs;

    RayBatchBuffer(ref<Device> pDevice, ShaderVar& var, uint32_t size, uint32_t numClusters)
        : size(size), numClusters(numClusters)
    {
        // Create buffers
        {
            pos = pDevice->createStructuredBuffer(
                var["allPos"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            dir = pDevice->createStructuredBuffer(
                var["allDir"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            normal = pDevice->createStructuredBuffer(
                var["allNormal"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            albedo = pDevice->createStructuredBuffer(
                var["allAlbedo"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            roughness = pDevice->createStructuredBuffer(
                var["allRoughness"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            vbuffer = pDevice->createStructuredBuffer(
                var["allVBuffer"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            color = pDevice->createStructuredBuffer(
                var["allColor"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
        }

        // Create diffuse pixel buffers
        {
            diffPos = pDevice->createStructuredBuffer(
                var["diffPos"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            diffDir = pDevice->createStructuredBuffer(
                var["diffDir"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            diffNormal = pDevice->createStructuredBuffer(
                var["diffNormal"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            diffAlbedo = pDevice->createStructuredBuffer(
                var["diffAlbedo"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            diffRoughness = pDevice->createStructuredBuffer(
                var["diffRoughness"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            diffActive = pDevice->createStructuredBuffer(
                var["diffActive"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            diffIndex = pDevice->createStructuredBuffer(
                var["diffIndex"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            diffColor = pDevice->createStructuredBuffer(
                var["diffColor"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
        }

        // Create specular pixel buffers
        {
            specPos = pDevice->createStructuredBuffer(
                var["specPos"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specDir = pDevice->createStructuredBuffer(
                var["specDir"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specNormal = pDevice->createStructuredBuffer(
                var["specNormal"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specAlbedo = pDevice->createStructuredBuffer(
                var["specAlbedo"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specRoughness = pDevice->createStructuredBuffer(
                var["specRoughness"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specVBuffer = pDevice->createStructuredBuffer(
                var["specVBuffer"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specActive = pDevice->createStructuredBuffer(
                var["specActive"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specIndex = pDevice->createStructuredBuffer(
                var["specIndex"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            specColor = pDevice->createStructuredBuffer(
                var["specColor"], size,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
        }

        // Create cluster buffers
        const uint32_t totalClusters = numClusters * size;
        {
            clusterPos = pDevice->createStructuredBuffer(
                var["clusterPos"], totalClusters,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            clusterDir = pDevice->createStructuredBuffer(
                var["clusterDir"], totalClusters,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            clusterScale = pDevice->createStructuredBuffer(
                var["clusterScale"], totalClusters,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
            clusterWeight = pDevice->createStructuredBuffer(
                var["clusterWeight"], totalClusters,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
                MemoryType::DeviceLocal,
                nullptr, false
            );
        }

        diffPtrs = {
            (float*) diffPos->getCudaMemory()->getMappedData(),
            (float*) diffDir->getCudaMemory()->getMappedData(),
            (float*) diffNormal->getCudaMemory()->getMappedData(),
            (float*) diffAlbedo->getCudaMemory()->getMappedData(),
            (float*) diffRoughness->getCudaMemory()->getMappedData(),
            nullptr, nullptr, nullptr, nullptr,
            (float*) diffColor->getCudaMemory()->getMappedData()
        };
        
        specPtrs = {
            (float*) specPos->getCudaMemory()->getMappedData(),
            (float*) specDir->getCudaMemory()->getMappedData(),
            (float*) specNormal->getCudaMemory()->getMappedData(),
            (float*) specAlbedo->getCudaMemory()->getMappedData(),
            (float*) specRoughness->getCudaMemory()->getMappedData(),

            (float*) clusterPos->getCudaMemory()->getMappedData(),
            (float*) clusterDir->getCudaMemory()->getMappedData(),
            (float*) clusterScale->getCudaMemory()->getMappedData(),
            (float*) clusterWeight->getCudaMemory()->getMappedData(),

            (float*) specColor->getCudaMemory()->getMappedData()
        };
    }
};


class NeuralRadiosity : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(NeuralRadiosity, "NeuralRadiosity", "Neural Radiosity Render Pass. (S. Hadadan, ToG 2021)");

    static ref<NeuralRadiosity> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<NeuralRadiosity>(pDevice, props);
    }

    NeuralRadiosity(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    void render(RenderContext* pRenderContext, const RenderData& renderData);
    void train(RenderContext* pRenderContext);
    // Screen rendering passes
    void firstSmoothPass(RenderContext* pRenderContext, const RenderData& renderData);
    void compactPass(RenderContext* pRenderContext, const RenderData& renderData);
    void resolvePass(RenderContext* pRenderContext, const RenderData& renderData);
    // Ray-batch rendering passes
    void randomSmooth(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);
    void sampleRHS(RenderContext* pRenderContext);
    void resolveRHS(RenderContext* pRenderContext);
    void compactBatch(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);
    void resolveBatch(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);
    // Common rendering passes
    void coneTrace(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);
    void modelInferenceCUDA(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);
    void modelTrainCUDA(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch);

    void updateFrameDim(const uint2 frameDim);
    void updatePrograms(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareResources(RenderContext* pRenderContext, const RenderData& renderData);
    void bindScreenData(ShaderVar& var, const RenderData& renderData, const std::string &name);
    void bindRayBatchData(ShaderVar& var, std::shared_ptr<RayBatchBuffer> pRayBatch, const std::string &name);
    DefineList getShaderDefines(const RenderData& renderData) const;
    bool loadTrainCamerasFromFile(const std::filesystem::path& path);
    void setCamera(uint32_t cameraIdx);

    // Internal state & parameters
    uint32_t mFrameCount = 0;
    uint2 mFrameDim = {};
    /// Shader variables changed flag
    bool mVarsChanged = false;
    /// Algorithm parameters
    uint32_t mNumSpecRays = 32;
    uint32_t mNumClusters = 4;
    uint32_t mNumKMeansIters = 10;
    /// Russian Roulette probability for random smooth.
    float mRSRRProb = 0.4f;
    /// Enable alpha test.
    bool mUseAlphaTest = true;
    /// Use environment lighting.
    bool mUseEnvLight = true;
    /// Adjust shading normals.
    bool mAdjustShadingNormals = true;
    /// Specular roughness threshold
    float mSpecularRoughnessThreshold = 0.5f;
    /// Force cull mode for all geometry, otherwise set it based on the scene.
    bool mForceCullMode = false;
    /// Cull mode to use for when mForceCullMode is true.
    RasterizerState::CullMode mCullMode = RasterizerState::CullMode::Back;
    /// UI variables
    RenderPassHelpers::IOSize mOutputSizeSelection = RenderPassHelpers::IOSize::Default;
    uint2 mFixedOutputSize = {512, 512};

    // Resources
    std::shared_ptr<RayBatchBuffer> mpRenderBatch;
    std::shared_ptr<RayBatchBuffer> mpTrainLHSBatch;
    std::shared_ptr<RayBatchBuffer> mpTrainRHSBatch;

    // Scene & Compute Passes
    ref<Scene> mpScene;
    ref<ComputePass> mpFirstSmoothPass;
    ref<ComputePass> mpCompactPass;
    ref<ComputePass> mpResolvePass;
    ref<ComputePass> mpRandomSmooth;
    ref<ComputePass> mpSampleRHS;
    ref<ComputePass> mpResolveRHS;
    ref<ComputePass> mpCompactBatch;
    ref<ComputePass> mpResolveBatch;
    ref<ComputePass> mpConeTrace;
    ref<SampleGenerator> mpSampleGenerator;
    std::unique_ptr<PrefixSum> mpPrefixSum;

    // CUDA
    ref<cuda_utils::CudaDevice> mpCudaDevice;
    std::unique_ptr<NRModel> mpNRModel;

    // Train mode parameters
    /// Render mode.
    bool mRenderMode = true;
    /// Training the model (Online training if both are true).
    bool mTrainMode = false;
    /// LHS and RHS
    uint32_t mBatchSize = 1u << 16;
    uint32_t mNumRHS = 32u;
    /// Cameras used for training.
    std::vector<ref<Camera>> mTrainCameras;
    std::filesystem::path mTrainCameraPath;
};
