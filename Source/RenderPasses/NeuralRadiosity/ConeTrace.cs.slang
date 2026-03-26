import Scene.Shading;
import Scene.RaytracingInline;
import PathState;
import RayTraceHelper;
import NetworkParams;

#include "Utils/Math/MathConstants.slangh"


RWStructuredBuffer<PackedHitInfo> specVBuffer;
RWStructuredBuffer<SpecNetInput> specInput;


#ifndef USE_ALPHA_TEST
#define USE_ALPHA_TEST true
#endif

#ifndef ADJUST_SHADING_NORMALS
#define ADJUST_SHADING_NORMALS true
#endif

#ifndef USE_ENV_LIGHT
#define USE_ENV_LIGHT true
#endif

#ifndef RAY_FLAGS
#define RAY_FLAGS true
#endif

#ifndef SPECULAR_ROUGHNESS_THRESHOLD
#define SPECULAR_ROUGHNESS_THRESHOLD 0.5f
#endif

#ifndef NUM_KMEANS_ITERS
#define NUM_KMEANS_ITERS 10
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


groupshared ShadingData sd;
groupshared IMaterialInstance mi;
groupshared BSDFProperties bsdfProperties;

groupshared float3 pos;
groupshared float3 dir;
groupshared float tanLobe;

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
    static const float kSpecularRoughnessThreshold = SPECULAR_ROUGHNESS_THRESHOLD;
    static const uint kRandomBase = 805459861u;
    static const uint waveSize = WAVE_SIZE;  // WaveGetLaneCount();

    static const uint numKMeansIters = NUM_KMEANS_ITERS;
    static const uint numSpecRays = NUM_SPEC_RAYS;
    static const uint numClusters = NUM_CLUSTERS;

    static const uint kLutSize = 32;
    static const float gTanLobeLUT[kLutSize] = {
        0.0,                    0.009716673754155636,   0.03886669501662254,    0.0874500647187233,
        0.15546678006649017,    0.24291685223579407,    0.3498002588748932,     0.4761170446872711,
        0.6218671202659607,     0.7870506048202515,     0.9716674089431763,     1.1757175922393799,
        1.3992010354995728,     1.642117977142334,      1.9044681787490845,     2.186251640319824,
        2.4874684810638428,     2.8081188201904297,     3.148202419281006,      3.5077192783355713,
        3.886669635772705,      4.285053253173828,      4.7028703689575195,     5.140120506286621,
        5.596804141998291,      6.072921276092529,      6.568471908569336,      7.083455562591553,
        7.617872714996338,      8.171723365783691,      8.745006561279297,      9.337723731994629
    };

    uint batchSize = 0;
    uint batchCount = 0;
    uint diffSize = 0;
    uint specSize = 0;

    float3 bboxMin;
    float3 bboxMax;

    void loadGroupData(const uint pixIdx, const uint threadIdx)
    {
        if (threadIdx == 0)
        {
            pos = specInput[pixIdx].getPos();
            dir = specInput[pixIdx].getDir();
            float roughness = specInput[pixIdx].getRoughness();
            HitInfo hit = unpackHitInfo(specVBuffer[pixIdx]);

            // Load shading data.
            sd = loadShadingData(hit, pos, dir);
            sd.posW = pos;

            // Disable diffuse lobes
            uint activeLobes = sd.mtl.getActiveLobes();
            activeLobes &= ~(uint)LobeType::Diffuse;
            sd.mtl.setActiveLobes(activeLobes);

            // Create material instance.
            let lod = ExplicitLodTextureSampler(0.f);
            let hints = getMaterialInstanceHints(hit, true, kAdjustShadingNormals);
            mi = gScene.materials.getMaterialInstance(sd, lod, hints);
            bsdfProperties = mi.getProperties(sd);

            // Calculate lobe info.
            dir = reflect(dir, sd.frame.N);
            float lutPos = roughness * kLutSize;
            uint lutIdx = (uint) lutPos;
            float frac = lutPos - lutIdx;
            tanLobe = lerp(gTanLobeLUT[lutIdx], gTanLobeLUT[lutIdx + 1], frac);
        }
        
        GroupMemoryBarrierWithGroupSync();
    }

    bool traceRandomDepth(const uint pixIdx, const uint threadIdx, out float rayT)
    {
        const uint rayIdx = pixIdx * numSpecRays + threadIdx;

        SampleGenerator sg = SampleGenerator(uint2(rayIdx & 0xFFFu, rayIdx >> 12), batchCount);

        BSDFSample result = {};
        bool valid = mi.sample(sd, sg, result);
        if (!valid) return false;

        SceneRayQuery<ConeTrace::kUseAlphaTest> sceneRayQuery;
        Ray ray;
        ray.origin = sd.posW;
        ray.dir = result.wo;
        ray.tMin = 0.f;
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
        uint waveIdx = threadIdx / waveSize;
        uint numWaves = (numSpecRays + waveSize - 1) / waveSize;

        // float to int
        static const float scale = 1e5f;
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
        if (threadIdx < numClusters)
        {
            gsVarsInt[threadIdx] = 0;
        }
        GroupMemoryBarrierWithGroupSync();

        // 3. Compute variance.
        int bestK = gsClusterIdx[threadIdx];
        int diff = rayTInt - gsCentersInt[bestK];
        int varsInt = (int) ((float) diff / scale * diff);
        InterlockedAdd(gsVarsInt[bestK], valid ? varsInt : 0);

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
            float mean = gsCenters[threadIdx];
            float size = gsCounts[threadIdx];
            float std = sqrt(gsVars[threadIdx] / max(size, 1.0f));
            // Calculate cluster info
            float3 clsPos = pos + mean * dir;
            float clsScale = (std + tanLobe * mean) / 2;
            float clsWeight = size / (float) numSpecRays;
            writeSpecInputClsBuffer(
                specInput, pixIdx, threadIdx,
                (clsPos - bboxMin) / (bboxMax - bboxMin),
                clsScale, clsWeight
            );
        }
        if (threadIdx == 0)
        {
            // Normalize to [0, 1]
            pos = (pos - bboxMin) / (bboxMax - bboxMin);
            specInput[pixIdx].data[SpecNetInput::posOffset + 0] = pos.x;
            specInput[pixIdx].data[SpecNetInput::posOffset + 1] = pos.y;
            specInput[pixIdx].data[SpecNetInput::posOffset + 2] = pos.z;
        }
    }

    void execute(const uint pixIdx, const uint threadIdx)
    {
        if (pixIdx >= specSize)
            return;

        // Load interaction.
        loadGroupData(pixIdx, threadIdx);

        float rayT;
        bool valid = traceRandomDepth(pixIdx, threadIdx, rayT);

        // K-Means
        kMeansInt(threadIdx, rayT, valid);

        // Write output
        writeOutput(pixIdx, threadIdx, rayT, valid);
    }
}


ParameterBlock<ConeTrace> gConeTrace;


[numthreads(NUM_SPEC_RAYS, 1, 1)]
void main(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID)
{
    gConeTrace.execute(groupID.x, groupThreadID.x);
}
