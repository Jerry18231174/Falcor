import Scene.Shading;
import Scene.RaytracingInline;
import PathState;
import LoadShadingData;

#include "Utils/Math/MathConstants.slangh"

// Input channels
RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<float4> gFSThp;
RWTexture2D<uint> gActive;

// Output channels
RWTexture2D<float4> gColor;
RWStructuredBuffer<float> gClusterMean;
RWStructuredBuffer<float> gClusterStd;
RWStructuredBuffer<uint> gClusterSize;

// G-Buffer channels
RWTexture2D<float4> gPos;
RWTexture2D<float4> gDir;
RWTexture2D<float4> gNormal;
RWTexture2D<float4> gAlbedo;
RWTexture2D<float> gRoughness;

#if !defined(USE_ALPHA_TEST) || !defined(ADJUST_SHADING_NORMALS) || !defined(USE_ENV_LIGHT) || !defined(RAY_FLAGS)
#error "Not all defines are set!"
#endif

#ifndef NUM_SPEC_RAYS
#define NUM_SPEC_RAYS 128
#endif

#ifndef NUM_CLUSTERS
#define NUM_CLUSTERS 4
#endif

#ifndef WAVE_SIZE
#define WAVE_SIZE 32
#endif


groupshared bool specPixel;
groupshared uint2 pixel;
groupshared ShadingData sd;
groupshared IMaterialInstance mi;
groupshared BSDFProperties bsdfProperties;

groupshared float3 dbgDir;

// Wave reduction buffers
groupshared float gsWaveMin[NUM_SPEC_RAYS / 32];
groupshared float gsWaveMax[NUM_SPEC_RAYS / 32];
groupshared float gsWaveSum[NUM_SPEC_RAYS / 32];
// K-Means
groupshared uint gsClusterIdx[NUM_SPEC_RAYS];
groupshared float gsCenters[NUM_CLUSTERS];
groupshared float gsVars[NUM_CLUSTERS];
groupshared uint gsCounts[NUM_CLUSTERS];
groupshared float gsNewCenters[NUM_CLUSTERS];

// Wave reduction buffers (int)
groupshared int gsWaveMinInt[NUM_SPEC_RAYS / 32];
groupshared int gsWaveMaxInt[NUM_SPEC_RAYS / 32];
groupshared int gsWaveSumInt[NUM_SPEC_RAYS / 32];
// K-Means (int)
groupshared int gsCentersInt[NUM_CLUSTERS];
groupshared int gsVarsInt[NUM_CLUSTERS];
groupshared int gsNewCentersInt[NUM_CLUSTERS];


float GroupActiveSum(float value, uint threadID, uint groupSize)
{
    uint laneIdx  = WaveGetLaneIndex();
    uint waveSize = WAVE_SIZE;  // WaveGetLaneCount();
    uint waveIdx  = threadID / waveSize;
    uint numWaves = (groupSize + waveSize - 1) / waveSize;

    float waveSum = WaveActiveSum(value);

    if (laneIdx == 0)
        gsWaveSum[waveIdx] = waveSum;

    GroupMemoryBarrierWithGroupSync();

    float groupSum = 0.0f;

    if (waveIdx == 0)
    {
        float val = (laneIdx < numWaves) ? gsWaveSum[laneIdx] : 0.0f;
        groupSum = WaveActiveSum(val);
    }

    GroupMemoryBarrierWithGroupSync();

    return groupSum;
}


struct ConeTrace
{
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const bool kAdjustShadingNormals = ADJUST_SHADING_NORMALS;
    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const uint kRayFlags = RAY_FLAGS;
    static const float kEnvMapDepth = 100000000.0f; // Arbitrary big number
    static const int kMaxRecursionDepth = 16;
    static const float kSpecularRoughnessThreshold = 0.5f;
    static const uint kRandomBase = 805459861u;

    static const uint numSpecRays = NUM_SPEC_RAYS;
    static const uint numClusters = NUM_CLUSTERS;

    uint2 frameDim = {};
    uint frameCount = 0;
    uint numKMeansIters = 0;

    void loadGroupData(const uint pixIdx, const uint threadIdx)
    {
        if (threadIdx == 0)
        {
            pixel = uint2(pixIdx % gConeTrace.frameDim.x, pixIdx / gConeTrace.frameDim.x);
            bool active = gActive[pixel] != 0;
            float roughness = gRoughness[pixel];

            // Trace postfix rays for specular interactions only.
            specPixel = active && (roughness <= kSpecularRoughnessThreshold);

            float3 pos = gPos[pixel].xyz;
            float3 dir = gDir[pixel].xyz;
            HitInfo hit = unpackHitInfo(gVBuffer[pixel]);

            // Load shading data.
            sd = loadShadingData(hit, pos, dir);

            // Disable diffuse lobes
            uint activeLobes = sd.mtl.getActiveLobes();
            activeLobes &= ~(uint)LobeType::Diffuse;
            sd.mtl.setActiveLobes(activeLobes);

            // Create material instance.
            let lod = ExplicitLodTextureSampler(0.f);
            let hints = getMaterialInstanceHints(hit, true, kAdjustShadingNormals);
            mi = gScene.materials.getMaterialInstance(sd, lod, hints);
            bsdfProperties = mi.getProperties(sd);
        }
        
        GroupMemoryBarrierWithGroupSync();
    }

    bool traceRandomDepth(const uint pixIdx, const uint threadIdx, out float rayT)
    {
        SampleGenerator sg = SampleGenerator(pixel, (frameCount * pixIdx + threadIdx) * kRandomBase);

        BSDFSample result;
        bool valid = mi.sample(sd, sg, result);
        if (!valid) return false;

        SceneRayQuery<ConeTrace::kUseAlphaTest> sceneRayQuery;
        Ray ray;
        ray.origin = sd.posW;
        ray.dir = result.wo;
        ray.tMin = 1e-4f;
        ray.tMax = 1e30f;

        HitInfo nextHit = sceneRayQuery.traceRay(ray, rayT, ConeTrace::kRayFlags);

        GroupMemoryBarrierWithGroupSync();

        return nextHit.isValid();
    }

    void kMeans(const uint threadIdx, float rayT, bool valid)
    {
        static_assert(numSpecRays <= 256, "Thread group size exceeds the square of warp size: 16^2");

        uint laneIdx = WaveGetLaneIndex();
        uint waveSize = WAVE_SIZE;  // WaveGetLaneCount();
        uint waveIdx = threadIdx / waveSize;
        uint numWaves = (numSpecRays + waveSize - 1) / waveSize;

        // 1.1 Min-max reduction inside a wave.
        float waveMin = WaveActiveMin(valid ? rayT : 0.0f);
        float waveMax = WaveActiveMax(valid ? rayT : -FLT_MAX);

        if (WaveIsFirstLane())
        {
            gsWaveMin[waveIdx] = waveMin;
            gsWaveMax[waveIdx] = waveMax;
        }
        GroupMemoryBarrierWithGroupSync();

        // 1.2 Min-max reduction over waves, then write to cluster centers.
        if (waveIdx == 0)
        {
            float bMin = 0.0f;
            float bMax = -FLT_MAX;

            if (laneIdx < numWaves)
            {
                bMin = gsWaveMin[laneIdx];
                bMax = gsWaveMax[laneIdx];
            }
            
            bMin = WaveActiveMin(bMin);
            bMax = WaveActiveMax(bMax);

            if (laneIdx < numClusters)
            {
                gsCenters[laneIdx] = bMin + (bMax - bMin) * laneIdx / (numClusters - 1);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 2. K-Means clustering.
        for (int iter = 0; iter < numKMeansIters; ++iter)
        {
            if (threadIdx < numClusters)
            {
                gsNewCenters[threadIdx] = 0.0f;
                gsCounts[threadIdx] = 0;
            }
            GroupMemoryBarrierWithGroupSync();

            int bestK = 0;
            float bestDist = fabs(rayT - gsCenters[0]);

            for (int k = 1; k < numClusters; ++k)
            {
                float dist = fabs(rayT - gsCenters[k]);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestK = k;
                }
            }
            gsClusterIdx[threadIdx] = bestK;

            InterlockedAdd(gsCounts[bestK], valid ? 1 : 0);
            // InterlockedAdd(gsNewCenters[bestK], valid ? rayT : 0.0f);
            for (int k = 0; k < numClusters; ++k)
            {
                bool inCluster = valid && (bestK == k);
                float sum = GroupActiveSum(inCluster ? rayT : 0.0f, threadIdx, numSpecRays);
            
                if (threadIdx == 0)
                    gsNewCenters[k] = sum;
            }
            GroupMemoryBarrierWithGroupSync();

            if (threadIdx < numClusters)
            {
                float count = max(gsCounts[threadIdx], 1.0f);
                gsCenters[threadIdx] = gsNewCenters[threadIdx] / count;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 3. Compute variance.
        int bestK = gsClusterIdx[threadIdx];
        float diff = rayT - gsCenters[bestK];
        // InterlockedAdd(gsVars[bestK], diff * diff);
        for (int k = 0; k < numClusters; ++k)
        {
            bool inCluster = valid && (bestK == k);
            float sum = GroupActiveSum(inCluster ? diff * diff : 0.0f, threadIdx, numSpecRays);
        
            if (threadIdx == 0)
                gsVars[k] = sum;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    void kMeansInt(const uint threadIdx, float rayT, bool valid)
    {
        static_assert(numSpecRays <= 256, "Thread group size exceeds the square of warp size: 16^2");

        uint laneIdx = WaveGetLaneIndex();
        uint waveSize = WAVE_SIZE;  // WaveGetLaneCount();
        uint waveIdx = threadIdx / waveSize;
        uint numWaves = (numSpecRays + waveSize - 1) / waveSize;

        // float to int
        float scale = 1e5f;
        int rayTInt = (int) (rayT * scale);

        // 1.1 Min-max reduction inside a wave.
        int waveMin = WaveActiveMin(valid ? rayTInt : INT32_MAX);
        int waveMax = WaveActiveMax(valid ? rayTInt : 0);

        if (WaveIsFirstLane())
        {
            gsWaveMinInt[waveIdx] = waveMin;
            gsWaveMaxInt[waveIdx] = waveMax;
        }
        GroupMemoryBarrierWithGroupSync();

        // 1.2 Min-max reduction over waves, then write to cluster centers.
        if (waveIdx == 0)
        {
            int bMin = INT32_MAX;
            int bMax = 0;

            if (laneIdx < numWaves)
            {
                bMin = gsWaveMinInt[laneIdx];
                bMax = gsWaveMaxInt[laneIdx];
            }
            
            bMin = WaveActiveMin(bMin);
            bMax = WaveActiveMax(bMax);

            if (laneIdx < numClusters)
            {
                gsCentersInt[laneIdx] = bMin + (bMax - bMin) * laneIdx / (numClusters - 1);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 2. K-Means clustering.
        for (int iter = 0; iter < numKMeansIters; ++iter)
        {
            if (threadIdx < numClusters)
            {
                gsNewCentersInt[threadIdx] = 0;
                gsCounts[threadIdx] = 0;
            }
            GroupMemoryBarrierWithGroupSync();

            int bestK = 0;
            int bestDist = abs(rayTInt - gsCentersInt[0]);

            for (int k = 1; k < numClusters; ++k)
            {
                int dist = abs(rayTInt - gsCentersInt[k]);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestK = k;
                }
            }
            gsClusterIdx[threadIdx] = bestK;

            InterlockedAdd(gsCounts[bestK], valid ? 1 : 0);
            InterlockedAdd(gsNewCentersInt[bestK], valid ? rayTInt : 0);
            GroupMemoryBarrierWithGroupSync();

            if (threadIdx < numClusters)
            {
                int count = max(gsCounts[threadIdx], 1);
                gsCentersInt[threadIdx] = gsNewCentersInt[threadIdx] / count;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // 3. Compute variance.
        int bestK = gsClusterIdx[threadIdx];
        int diff = rayTInt - gsCentersInt[bestK];
        InterlockedAdd(gsVarsInt[bestK], diff * diff);

        GroupMemoryBarrierWithGroupSync();

        if (threadIdx < numClusters)
        {
            gsVars[threadIdx] = gsVarsInt[threadIdx] / scale;
            gsCenters[threadIdx] = gsCentersInt[threadIdx] / scale;
        }
    }

    void writeOutput(const uint pixIdx, const uint threadIdx, float rayT, bool valid)
    {
        if (threadIdx < numClusters)
        {
            uint index = pixIdx * numClusters + threadIdx;
            float count = max(gsCounts[threadIdx], 1.0f);
            gClusterStd[index] = sqrt(gsVars[threadIdx] / count);
            gClusterMean[index] = gsCenters[threadIdx];
            gClusterSize[index] = gsCounts[threadIdx];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    void execute(const uint pixIdx, const uint threadIdx)
    {
        if (pixIdx >= (gConeTrace.frameDim.x * gConeTrace.frameDim.y))
            return;

        pixel = uint2(pixIdx % gConeTrace.frameDim.x, pixIdx / gConeTrace.frameDim.x);

        // Load interaction.
        loadGroupData(pixIdx, threadIdx);

        // Trace postfix rays for specular interactions only.
        if (!specPixel) return;

        float rayT;
        bool valid = traceRandomDepth(pixIdx, threadIdx, rayT);

        // K-Means
        kMeansInt(threadIdx, rayT, valid);

        // Write output
        writeOutput(pixIdx, threadIdx, rayT, valid);

        if (threadIdx == 0)
        {
            // bool empty = gClusterSize[pixIdx * 4] == 0;
            // gFSThp[pixel] = float4(empty ? 0.0f : (gClusterMean[pixIdx * 4]) / 5.0f, empty ? 1.0f : 0.0f, 0.0f, 0.0f);
            gColor[pixel] = float4((gClusterMean[pixIdx * 4]) / 1.0f);
            // gColor[pixel] = float4(dbgDir, 0.0f);
        }
    }
}


ParameterBlock<ConeTrace> gConeTrace;


[numthreads(NUM_SPEC_RAYS, 1, 1)]
void main(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID)
{
    gConeTrace.execute(groupID.x, groupThreadID.x);
}