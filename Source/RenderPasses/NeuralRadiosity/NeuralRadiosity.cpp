#include "NeuralRadiosity.h"

#include "Utils/Math/VectorJson.h"

#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>

namespace
{
    // Render pass inputs and outputs.

    const Falcor::ChannelList kInputChannels =
    {
        // { kInputVBuffer,        "gVBuffer",         "Visibility buffer in packed format" },
        // { kInputViewDir,        "gViewW",           "World-space view direction (xyz float format)", true /* optional */ },
    };

    const std::string kOutputColor = "color";
    const std::string kOutputVBuffer = "vbuffer";
    const std::string kOutputActive = "active";
    const std::string kOutputPos = "pos";
    const std::string kOutputDir = "dir";
    const std::string kOutputNormal = "normal";
    const std::string kOutputAlbedo = "albedo";
    const std::string kOutputRoughness = "roughness";

    const Falcor::ChannelList kOutputChannels =
    {
        { kOutputColor,         "gColor",       "Output color",                       false,  ResourceFormat::RGBA32Float },
        { kOutputVBuffer,       "gVBuffer",     "Visibility buffer in packed format", false,  ResourceFormat::RGBA32Uint },
        { kOutputActive,        "gActive",      "If the pixel is active",             false,  ResourceFormat::R32Uint },
        { kOutputPos,           "gPos",         "Output position",                    false,  ResourceFormat::RGBA32Float },
        { kOutputDir,           "gDir",         "Output direction",                   false,  ResourceFormat::RGBA32Float },
        { kOutputNormal,        "gNormal",      "Output normal",                      false,  ResourceFormat::RGBA32Float },
        { kOutputAlbedo,        "gAlbedo",      "Output albedo",                      false,  ResourceFormat::RGBA32Float },
        { kOutputRoughness,     "gRoughness",   "Output roughness",                   false,  ResourceFormat::R32Float },
    };

    // Program files.
    const std::string kFirstSmoothPassFile = "RenderPasses/NeuralRadiosity/FirstSmoothPass.cs.slang";
    const std::string kCompactPassFile = "RenderPasses/NeuralRadiosity/CompactPass.cs.slang";
    const std::string kResolvePassFile = "RenderPasses/NeuralRadiosity/ResolvePass.cs.slang";

    const std::string kRandomSmoothFile = "RenderPasses/NeuralRadiosity/RandomSmooth.cs.slang";
    const std::string kSampleRHSFile = "RenderPasses/NeuralRadiosity/SampleRHS.cs.slang";
    const std::string kResolveRHSFile = "RenderPasses/NeuralRadiosity/ResolveRHS.cs.slang";
    const std::string kCompactBatchFile = "RenderPasses/NeuralRadiosity/CompactBatch.cs.slang";
    const std::string kResolveBatchFile = "RenderPasses/NeuralRadiosity/ResolveBatch.cs.slang";
    
    const std::string kConeTraceFile = "RenderPasses/NeuralRadiosity/ConeTrace.cs.slang";

    const FileDialogFilterVec kCameraJsonFilters = {
        {"json", "JSON Files"},
    };

    float3 parseFloat3(const nlohmann::json& value, std::string_view key)
    {
        if (!value.is_array() || value.size() != 3)
        {
            FALCOR_THROW("Camera field '{}' must be an array of 3 numbers.", key);
        }

        return value.get<float3>();
    }
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, NeuralRadiosity>();
}

NeuralRadiosity::NeuralRadiosity(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    // parseProperties();

    // Create random engine
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_DEFAULT);
}

Properties NeuralRadiosity::getProperties() const
{
    return {};
}

RenderPassReflection NeuralRadiosity::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels, ResourceBindFlags::UnorderedAccess, sz);

    return reflector;
}

void NeuralRadiosity::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // renderData holds the requested resources
    const auto& pOutput = renderData.getTexture(kOutputColor);
    FALCOR_ASSERT(pOutput);

    // Set output frame dimension.
    updateFrameDim(uint2(pOutput->getWidth(), pOutput->getHeight()));

    // If there is no scene, clear the output and return.
    if (mpScene == nullptr)
    {
        clearRenderPassChannels(pRenderContext, kOutputChannels, renderData);
        return;
    }

    // Update shader program specialization.
    updatePrograms(pRenderContext, renderData);

    // Prepare resources.
    prepareResources(pRenderContext, renderData);

    // Render the scene.
    if (mRenderMode == RenderMode::Render || mRenderMode == RenderMode::OnlineTrain)
    {
        render(pRenderContext, renderData);
    }

    // Train the model.
    if (mRenderMode == RenderMode::Train || mRenderMode == RenderMode::OnlineTrain)
    {
        train(pRenderContext);
    }

    mFrameCount++;
    mVarsChanged = false;
}

void NeuralRadiosity::renderUI(Gui::Widgets& widget)
{
    if (widget.dropdown("Render Mode", mRenderMode))
    {
        mVarsChanged = true;
        if (mRenderMode == RenderMode::Train)
        {
            if (mNumSpecRays != NUM_SPEC_RAYS_TRAIN || mNumKMeansIters != NUM_KMEANS_ITERS_TRAIN)
            {
                mNumSpecRays = NUM_SPEC_RAYS_TRAIN;
                mNumKMeansIters = NUM_KMEANS_ITERS_TRAIN;
                mpConeTrace = nullptr;
            }
        }
        else if (mRenderMode == RenderMode::Render || mRenderMode == RenderMode::OnlineTrain)
        {
            if (mNumSpecRays != NUM_SPEC_RAYS_RENDER || mNumKMeansIters != NUM_KMEANS_ITERS_RENDER)
            {
                mNumSpecRays = NUM_SPEC_RAYS_RENDER;
                mNumKMeansIters = NUM_KMEANS_ITERS_RENDER;
                mpConeTrace = nullptr;
            }
        }
    }

    if (widget.button("Load train cameras"))
    {
        std::filesystem::path path;
        if (openFileDialog(kCameraJsonFilters, path))
        {
            loadTrainCamerasFromFile(path);
            mVarsChanged = true;
        }
    }

    widget.text(fmt::format("Train batchsize: {}, RHS samples: {}", mBatchSize, mNumRHS));
    widget.text(fmt::format("Num spec rays: {}, clusters: {}, KMeans iters: {}", mNumSpecRays, mNumClusters, mNumKMeansIters));
    widget.text(fmt::format("Train cameras: {}", mTrainCameras.size()));
    widget.text(mTrainCameraPath.empty() ? "Camera file: N/A" : fmt::format("Camera file: {}", mTrainCameraPath.filename().string()));

    if (!mTrainCameraPath.empty())
    {
        widget.tooltip(mTrainCameraPath.string());
    }
}

void NeuralRadiosity::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpFirstSmoothPass = nullptr;
    mpConeTrace = nullptr;
    mpResolvePass = nullptr;
    mVarsChanged = true;
}

// Private methods

void NeuralRadiosity::render(RenderContext* pRenderContext, const RenderData& renderData)
{
    // First smooth pass.
    firstSmoothPass(pRenderContext, renderData);
    // Compact pass.
    compactPass(pRenderContext, renderData);
    // Cone tracing pass.
    coneTrace(pRenderContext, mpRenderBatch);
    // Model querying: CUDA kernel.
    modelInferenceCUDA(pRenderContext, mpRenderBatch);
    // Resolve pass.
    resolvePass(pRenderContext, renderData);
}

void NeuralRadiosity::train(RenderContext* pRenderContext)
{
    // Sample random smooth points (LHS)
    randomSmooth(pRenderContext, mpTrainLHSBatch);
    // Sample RHS
    sampleRHS(pRenderContext);

    // Render RHS
    {
        // 1. Compaction
        compactBatch(pRenderContext, mpTrainRHSBatch);
        // 2. Cone tracing
        coneTrace(pRenderContext, mpTrainRHSBatch);
        // 3. Model inference
        modelInferenceCUDA(pRenderContext, mpTrainRHSBatch);
        // 4. Resolve
        resolveBatch(pRenderContext, mpTrainRHSBatch);
    }

    // Train LHS with RHS results
    {
        // 1. Compaction
        compactBatch(pRenderContext, mpTrainLHSBatch);
        // 2. Cone tracing
        coneTrace(pRenderContext, mpTrainLHSBatch);
        // 3. Resolve RHS color (write to lhs->diffColor/specColor)
        resolveRHS(pRenderContext);
        // 4. Model forward (target should be written to diffColor/specColor)
        modelTrainCUDA(pRenderContext, mpTrainLHSBatch);
    }
}

void NeuralRadiosity::firstSmoothPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "FirstSmoothPass");

    ShaderVar var = mpFirstSmoothPass->getRootVar();
    bindScreenData(var, renderData, "gFirstSmoothPass");

    var["diffActive"] = mpRenderBatch->diffActive;
    var["diffIndex"] = mpRenderBatch->diffIndex;
    var["specActive"] = mpRenderBatch->specActive;
    var["specIndex"] = mpRenderBatch->specIndex;

    mpFirstSmoothPass->execute(pRenderContext, uint3(mFrameDim, 1));
}

void NeuralRadiosity::compactPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "CompactPass");

    // Prefix sum from active mask to index
    mpPrefixSum->execute(
        pRenderContext,
        mpRenderBatch->diffIndex,
        mFrameDim.x * mFrameDim.y,
        &mpRenderBatch->diffSize
    );

    mpPrefixSum->execute(
        pRenderContext,
        mpRenderBatch->specIndex,
        mFrameDim.x * mFrameDim.y,
        &mpRenderBatch->specSize
    );

    ShaderVar var = mpCompactPass->getRootVar();
    const auto name = "gCompactPass";
    bindScreenData(var, renderData, name);

    const AABB& aabb = mpScene->getSceneBounds();
    var[name]["bboxMin"] = aabb.minPoint;
    var[name]["bboxMax"] = aabb.maxPoint;

    var[name]["diffSize"] = mpRenderBatch->diffSize;
    var[name]["specSize"] = mpRenderBatch->specSize;

    var["diffActive"] = mpRenderBatch->diffActive;
    var["diffIndex"] = mpRenderBatch->diffIndex;
    var["diffInput"] = mpRenderBatch->diffInput;

    var["specActive"] = mpRenderBatch->specActive;
    var["specIndex"] = mpRenderBatch->specIndex;
    var["specInput"] = mpRenderBatch->specInput;
    var["specVBuffer"] = mpRenderBatch->specVBuffer;

    mpCompactPass->execute(pRenderContext, uint3(mFrameDim, 1));
}

void NeuralRadiosity::resolvePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ResolvePass");

    ShaderVar var = mpResolvePass->getRootVar();
    const auto name = "gResolvePass";
    bindScreenData(var, renderData, name);

    var[name]["diffSize"] = mpRenderBatch->diffSize;
    var[name]["specSize"] = mpRenderBatch->specSize;

    var["diffActive"] = mpRenderBatch->diffActive;
    var["diffIndex"] = mpRenderBatch->diffIndex;
    var["diffColor"] = mpRenderBatch->diffColor;

    var["specActive"] = mpRenderBatch->specActive;
    var["specIndex"] = mpRenderBatch->specIndex;
    var["specColor"] = mpRenderBatch->specColor;
    var["specVBuffer"] = mpRenderBatch->specVBuffer;

    if (mRenderMode == RenderMode::Train || mRenderMode == RenderMode::OnlineTrain)
        var["debug"] = mpTrainRHSBatch->specColor;

    mpResolvePass->execute(pRenderContext, uint3(mFrameDim, 1));
}

void NeuralRadiosity::randomSmooth(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    // Sample a random camera (offline training)
    if (mRenderMode == RenderMode::Train && (mTrainCameras.size() > 0))
    {
        std::mt19937 rng(mFrameCount);
        std::uniform_int_distribution<uint32_t> dist(0, mTrainCameras.size() - 1);
        setCamera(dist(rng));
    }

    ShaderVar var = mpRandomSmooth->getRootVar();
    bindRayBatchData(var, pRayBatch, "gRandomSmooth");

    mpRandomSmooth->execute(pRenderContext, uint3(pRayBatch->size, 1, 1));
}

void NeuralRadiosity::sampleRHS(RenderContext* pRenderContext)
{
    ShaderVar var = mpSampleRHS->getRootVar();
    const auto name = "gSampleRHS";

    // Bind parameters
    var[name]["batchSize"] = mBatchSize;
    var[name]["batchCount"] = mFrameCount;

    // LHS buffers
    {
        var["lhsPos"] = mpTrainLHSBatch->pos;
        var["lhsDir"] = mpTrainLHSBatch->dir;
        var["lhsVBuffer"] = mpTrainLHSBatch->vbuffer;
        var["lhsDiffActive"] = mpTrainLHSBatch->diffActive;
        var["lhsSpecActive"] = mpTrainLHSBatch->specActive;
    }

    // RHS buffers
    {
        var["rhsPos"] = mpTrainRHSBatch->pos;
        var["rhsDir"] = mpTrainRHSBatch->dir;
        var["rhsNormal"] = mpTrainRHSBatch->normal;
        var["rhsAlbedo"] = mpTrainRHSBatch->albedo;
        var["rhsRoughness"] = mpTrainRHSBatch->roughness;
        var["rhsVBuffer"] = mpTrainRHSBatch->vbuffer;
        var["rhsColor"] = mpTrainRHSBatch->color;

        var["rhsDiffActive"] = mpTrainRHSBatch->diffActive;
        var["rhsDiffIndex"] = mpTrainRHSBatch->diffIndex;
        var["rhsSpecActive"] = mpTrainRHSBatch->specActive;
        var["rhsSpecIndex"] = mpTrainRHSBatch->specIndex;
    }

    mpSampleRHS->execute(pRenderContext, uint3(mBatchSize * mNumRHS, 1, 1));
}

void NeuralRadiosity::resolveRHS(RenderContext* pRenderContext)
{
    ShaderVar var = mpResolveRHS->getRootVar();
    const auto name = "gResolveRHS";
    
    bindRayBatchData(var, mpTrainLHSBatch, name);
    var["rhsColor"] = mpTrainRHSBatch->color;

    mpResolveRHS->execute(pRenderContext, uint3(mBatchSize, 1, 1));
}

void NeuralRadiosity::compactBatch(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    // Prefix sum from active mask to index
    mpPrefixSum->execute(
        pRenderContext,
        pRayBatch->diffIndex,
        pRayBatch->size,
        &pRayBatch->diffSize
    );

    mpPrefixSum->execute(
        pRenderContext,
        pRayBatch->specIndex,
        pRayBatch->size,
        &pRayBatch->specSize
    );

    ShaderVar var = mpCompactBatch->getRootVar();
    const auto name = "gCompactBatch";
    bindRayBatchData(var, pRayBatch, name);

    const AABB& aabb = mpScene->getSceneBounds();
    var[name]["bboxMin"] = aabb.minPoint;
    var[name]["bboxMax"] = aabb.maxPoint;

    mpCompactBatch->execute(pRenderContext, uint3(pRayBatch->size, 1, 1));
}

void NeuralRadiosity::resolveBatch(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    ShaderVar var = mpResolveBatch->getRootVar();
    bindRayBatchData(var, pRayBatch, "gResolveBatch");

    mpResolveBatch->execute(pRenderContext, uint3(pRayBatch->size, 1, 1));
}

void NeuralRadiosity::coneTrace(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    FALCOR_PROFILE(pRenderContext, "coneTrace");

    ShaderVar var = mpConeTrace->getRootVar();
    const auto name = "gConeTrace";
    // Bind parameters
    var[name]["batchSize"] = pRayBatch->size;
    var[name]["batchCount"] = mFrameCount;
    var[name]["diffSize"] = pRayBatch->diffSize;
    var[name]["specSize"] = pRayBatch->specSize;

    const AABB& aabb = mpScene->getSceneBounds();
    var[name]["bboxMin"] = aabb.minPoint;
    var[name]["bboxMax"] = aabb.maxPoint;

    var["specVBuffer"] = pRayBatch->specVBuffer;
    var["specInput"] = pRayBatch->specInput;

    mpConeTrace->execute(pRenderContext, uint3(pRayBatch->specSize * mNumSpecRays, 1, 1));
}

void NeuralRadiosity::modelInferenceCUDA(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    FALCOR_ASSERT(pRenderContext);
    FALCOR_PROFILE(pRenderContext, "modelInferenceCUDA");

    // Synchronize Falcor->CUDA before touching the shared buffer on CUDA.
    pRenderContext->waitForFalcor(mpNRModel->stream());

    mpNRModel->inference(pRayBatch->getDiffPtrs(), pRayBatch->getSpecPtrs());

    // Synchronize CUDA->Falcor so following passes see CUDA writes.
    pRenderContext->waitForCuda(mpNRModel->stream());
}

void NeuralRadiosity::modelTrainCUDA(RenderContext* pRenderContext, std::shared_ptr<RayBatchBuffer> pRayBatch)
{
    FALCOR_ASSERT(pRenderContext);

    // Synchronize Falcor->CUDA before touching the shared buffer on CUDA.
    pRenderContext->waitForFalcor(mpNRModel->stream());

    mpNRModel->train(pRayBatch->getDiffPtrs(), pRayBatch->getSpecPtrs());
    
    // Synchronize CUDA->Falcor so following passes see CUDA writes.
    pRenderContext->waitForCuda(mpNRModel->stream());
}

void NeuralRadiosity::updateFrameDim(const uint2 frameDim)
{
    FALCOR_ASSERT(frameDim.x > 0 && frameDim.y > 0);

    if (any(mFrameDim != frameDim))
    {
        mVarsChanged = true;
    }
    mFrameDim = frameDim;
}

void NeuralRadiosity::updatePrograms(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mVarsChanged) return;
    const auto startTime = std::chrono::steady_clock::now();

    if (!mpPrefixSum)
    {
        mpPrefixSum = std::make_unique<PrefixSum>(mpDevice);
    }

    ProgramDesc baseDesc;
    baseDesc.addShaderModules(mpScene->getShaderModules());
    baseDesc.addTypeConformances(mpScene->getTypeConformances());

    DefineList defines;
    defines.add(mpScene->getSceneDefines());
    defines.add(mpSampleGenerator->getDefines());
    defines.add(getShaderDefines(renderData));

    if (!mpFirstSmoothPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kFirstSmoothPassFile).csEntry("main");

        mpFirstSmoothPass = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpFirstSmoothPass->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpCompactPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kCompactPassFile).csEntry("main");

        mpCompactPass = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpCompactPass->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpResolvePass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kResolvePassFile).csEntry("main");

        mpResolvePass = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpResolvePass->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    const auto passEndTime = std::chrono::steady_clock::now();
    const auto passElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(passEndTime - startTime).count();
    logInfo("Screen passes created: \t{} ms", passElapsedMs);

    if (!mpRandomSmooth)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kRandomSmoothFile).csEntry("main");

        mpRandomSmooth = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpRandomSmooth->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpSampleRHS)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kSampleRHSFile).csEntry("main");

        mpSampleRHS = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpSampleRHS->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpResolveRHS)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kResolveRHSFile).csEntry("main");

        mpResolveRHS = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpResolveRHS->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpCompactBatch)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kCompactBatchFile).csEntry("main");

        mpCompactBatch = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpCompactBatch->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpResolveBatch)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kResolveBatchFile).csEntry("main");

        mpResolveBatch = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpResolveBatch->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    if (!mpConeTrace)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kConeTraceFile).csEntry("main");

        mpConeTrace = ComputePass::create(mpDevice, desc, defines, true);

        // Bind static resources
        ShaderVar var = mpConeTrace->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
    }

    const auto batchEndTime = std::chrono::steady_clock::now();
    const auto batchElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(batchEndTime - passEndTime).count();
    logInfo("Batch passes created: \t{} ms", batchElapsedMs);

    if (!mpNRModel)
    {
        mpNRModel = std::make_unique<NRModel>();
    }

    const auto modelEndTime = std::chrono::steady_clock::now();
    const auto modelElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(modelEndTime - batchEndTime).count();
    logInfo("NRModel created: \t{} ms", modelElapsedMs);
    logInfo("NeuralRadiosity::updatePrograms() took {} ms", passElapsedMs + batchElapsedMs + modelElapsedMs);
}

void NeuralRadiosity::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mVarsChanged) return;
    const auto startTime = std::chrono::steady_clock::now();

    ShaderVar var = mpRandomSmooth->getRootVar();

    if (!mpRenderBatch)
        mpRenderBatch = std::make_shared<RayBatchBuffer>(mpDevice, var, mFrameDim.x * mFrameDim.y, mNumClusters, "RenderBatch", false);
    if (!mpTrainLHSBatch)
        mpTrainLHSBatch = std::make_shared<RayBatchBuffer>(mpDevice, var, mBatchSize, mNumClusters, "LHSBatch", true);
    if (!mpTrainRHSBatch)
        mpTrainRHSBatch = std::make_shared<RayBatchBuffer>(mpDevice, var, mBatchSize * mNumRHS, mNumClusters, "RHSBatch", true);

    if (mRenderMode == RenderMode::Render || mRenderMode == RenderMode::OnlineTrain)
    {
        mpRenderBatch->resize(mFrameDim.x * mFrameDim.y, mNumClusters);
    }
    if (mRenderMode == RenderMode::Train || mRenderMode == RenderMode::OnlineTrain)
    {
        mpTrainLHSBatch->resize(mBatchSize, mNumClusters);
        mpTrainRHSBatch->resize(mBatchSize * mNumRHS, mNumClusters);
    }

    const auto endTime = std::chrono::steady_clock::now();
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    logInfo("NeuralRadiosity::prepareResources() took {} ms", elapsedMs);
}

void NeuralRadiosity::bindScreenData(ShaderVar& var, const RenderData& renderData, const std::string& name)
{
    // Bind parameters
    var[name]["frameDim"] = mFrameDim;
    var[name]["frameCount"] = mFrameCount;
    // Bind input & output textures
    for (const auto& channel : kInputChannels)
        var[channel.texname] = renderData.getTexture(channel.name);
    for (const auto& channel : kOutputChannels)
        var[channel.texname] = renderData.getTexture(channel.name);
}

void NeuralRadiosity::bindRayBatchData(ShaderVar& var, std::shared_ptr<RayBatchBuffer> pRayBatch, const std::string &name)
{
    // Bind parameters
    var[name]["batchSize"] = pRayBatch->size;
    var[name]["batchCount"] = mFrameCount;
    var[name]["diffSize"] = pRayBatch->diffSize;
    var[name]["specSize"] = pRayBatch->specSize;
    
    // Bind buffers
    var["allPos"] = pRayBatch->pos;
    var["allDir"] = pRayBatch->dir;
    var["allNormal"] = pRayBatch->normal;
    var["allAlbedo"] = pRayBatch->albedo;
    var["allRoughness"] = pRayBatch->roughness;
    var["allVBuffer"] = pRayBatch->vbuffer;
    var["allColor"] = pRayBatch->color;

    // Bind diffuse buffers
    var["diffActive"] = pRayBatch->diffActive;
    var["diffIndex"] = pRayBatch->diffIndex;
    var["diffInput"] = pRayBatch->diffInput;
    var["diffColor"] = pRayBatch->diffColor;

    // Bind specular buffers
    var["specActive"] = pRayBatch->specActive;
    var["specIndex"] = pRayBatch->specIndex;
    var["specInput"] = pRayBatch->specInput;
    var["specColor"] = pRayBatch->specColor;
    var["specVBuffer"] = pRayBatch->specVBuffer;
}

DefineList NeuralRadiosity::getShaderDefines(const RenderData& renderData) const
{
    DefineList defines;

    defines.add("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");
    defines.add("ADJUST_SHADING_NORMALS", mAdjustShadingNormals ? "1" : "0");
    defines.add("USE_ENV_LIGHT", mUseEnvLight ? "1" : "0");
    defines.add("SPECULAR_ROUGHNESS_THRESHOLD", std::to_string(mSpecularRoughnessThreshold));
    // Add cone trace parameters
    defines.add("NUM_KMEANS_ITERS", std::to_string(mNumKMeansIters));
    defines.add("NUM_SPEC_RAYS", std::to_string(mNumSpecRays));
    defines.add("NUM_CLUSTERS", std::to_string(mNumClusters));
    // Add sample RHS parameters
    defines.add("NUM_RHS", std::to_string(mNumRHS));
    // Add random smooth parameters
    defines.add("RS_RR_PROB", std::to_string(mRSRRProb));
    // Setup ray flags.
    RayFlags rayFlags = RayFlags::None;
    if (mForceCullMode && mCullMode == RasterizerState::CullMode::Front)
        rayFlags = RayFlags::CullFrontFacingTriangles;
    else if (mForceCullMode && mCullMode == RasterizerState::CullMode::Back)
        rayFlags = RayFlags::CullBackFacingTriangles;
    defines.add("RAY_FLAGS", std::to_string((uint32_t)rayFlags));

    // Set 'is_valid_<name>' defines to inform the program of which ones it can access.
    defines.add(getValidResourceDefines(kInputChannels, renderData));
    defines.add(getValidResourceDefines(kOutputChannels, renderData));
    return defines;
}

bool NeuralRadiosity::loadTrainCamerasFromFile(const std::filesystem::path& path)
{
    try
    {
        std::ifstream ifs(path);
        if (!ifs)
        {
            FALCOR_THROW("Failed to open file '{}'.", path.string());
        }

        const nlohmann::json json = nlohmann::json::parse(ifs);
        if (!json.is_array())
        {
            FALCOR_THROW("Camera file '{}' must contain a JSON array.", path.string());
        }

        std::vector<ref<Camera>> cameras;
        cameras.reserve(json.size());

        for (size_t i = 0; i < json.size(); ++i)
        {
            const auto& entry = json.at(i);
            if (!entry.is_object())
            {
                FALCOR_THROW("Camera entry {} must be a JSON object.", i);
            }

            auto camera = Camera::create(fmt::format("train_camera_{}", i));
            camera->setPosition(parseFloat3(entry.at("position"), "position"));
            camera->setTarget(parseFloat3(entry.at("target"), "target"));
            camera->setUpVector(parseFloat3(entry.at("up"), "up"));
            camera->setFocalLength(entry.at("focalLength").get<float>());
            camera->setFocalDistance(entry.at("focalDistance").get<float>());
            camera->setApertureRadius(entry.at("apertureRadius").get<float>());

            if (entry.contains("aspectRatio")) camera->setAspectRatio(entry.at("aspectRatio").get<float>());
            if (entry.contains("nearPlane")) camera->setNearPlane(entry.at("nearPlane").get<float>());
            if (entry.contains("farPlane")) camera->setFarPlane(entry.at("farPlane").get<float>());

            cameras.push_back(camera);
        }

        mTrainCameras = std::move(cameras);
        mTrainCameraPath = path;
        logInfo("Loaded {} training cameras from '{}'.", mTrainCameras.size(), path.string());
        return true;
    }
    catch (const std::exception& e)
    {
        logError("Failed to load training cameras from '{}': {}", path.string(), e.what());
        return false;
    }
}

void NeuralRadiosity::setCamera(uint32_t cameraIdx)
{
    if (!mpScene)
    {
        logWarning("NeuralRadiosity::setCamera() called without an active scene.");
        return;
    }

    if (cameraIdx >= mTrainCameras.size())
    {
        logWarning("NeuralRadiosity::setCamera() camera index {} is out of range ({} cameras loaded).", cameraIdx, mTrainCameras.size());
        return;
    }

    const auto& pSrcCamera = mTrainCameras[cameraIdx];
    const auto& pDstCamera = mpScene->getCamera();
    if (!pSrcCamera || !pDstCamera)
    {
        logWarning("NeuralRadiosity::setCamera() source or destination camera is null.");
        return;
    }

    pDstCamera->setPosition(pSrcCamera->getPosition());
    pDstCamera->setTarget(pSrcCamera->getTarget());
    pDstCamera->setUpVector(pSrcCamera->getUpVector());
    pDstCamera->setAspectRatio(pSrcCamera->getAspectRatio());
    pDstCamera->setFocalLength(pSrcCamera->getFocalLength());
    pDstCamera->setFrameHeight(pSrcCamera->getFrameHeight());
    pDstCamera->setFocalDistance(pSrcCamera->getFocalDistance());
    pDstCamera->setApertureRadius(pSrcCamera->getApertureRadius());
    pDstCamera->setShutterSpeed(pSrcCamera->getShutterSpeed());
    pDstCamera->setISOSpeed(pSrcCamera->getISOSpeed());
    pDstCamera->setNearPlane(pSrcCamera->getNearPlane());
    pDstCamera->setFarPlane(pSrcCamera->getFarPlane());
}
