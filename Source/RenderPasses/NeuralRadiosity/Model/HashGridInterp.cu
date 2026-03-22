#include "HashGridInterp.h"
#include <vector>
#include <cuda_runtime.h>
#include <assert.h>


__constant__ uint32_t cResolutions[MAX_LEVELS];
__constant__ uint32_t cGridSizes[MAX_LEVELS];
__constant__ uint32_t cGridOffsets[MAX_LEVELS];
__constant__ float cGridScales[MAX_LEVELS];


__device__ __forceinline__ uint3 _corner_offset(int c) {
    return make_uint3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
}

template<int LOG2_HASHMAP_SIZE>
__device__ __forceinline__ uint32_t _hash_index(uint3 index, int level) {
    constexpr uint32_t MAX_HASH = 1 << LOG2_HASHMAP_SIZE;

    uint32_t res1 = cResolutions[level] + 1;
    uint32_t denseMax = cGridSizes[level];

    uint32_t hashed = 0;

    if (denseMax > MAX_HASH) {
        // Hash mode
        uint64_t result =
            index.x * PRIME_X +
            index.y * PRIME_Y +
            index.z * PRIME_Z;
        hashed = static_cast<uint32_t>(result & (MAX_HASH - 1));
    } else {
        // Dense indexing mode
        // TODO: remove mod after applying bbox normalize
        hashed = ((res1 * index.x + index.y) * res1 + index.z) % denseMax;
    }

    return hashed;
}

template<int N_LEVELS, int LOG2_HASHMAP_SIZE, int N_FEATURES_PER_LEVEL>
__global__ void forwardKernel(
    const float* grids,
    float* clsInput,
    uint32_t count,
    uint32_t fullDim
) {
    uint32_t pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixIdx >= count) return;
    
    float* pixOut = clsInput + pixIdx * fullDim;

    const float *pixIn = clsInput + pixIdx * fullDim + N_FEATURES_PER_LEVEL;
    float3 pos3 = make_float3(pixIn[0], pixIn[1], pixIn[2]);
    float scaleVal = pixIn[6];

    // Get corresponding level and layer interpolation weight
    // Lower level means coarser, larger voxel size
    int upperLevel, lowerLevel;
    float upperLayerWeight = 0.0f;
    float lowerLayerWeight = 0.0f;

    {   // Upper(finer) level, fit bottom-up
        int level = N_LEVELS;
        for (int i = 0; i < N_LEVELS; ++i) {
            float voxelScale = cGridScales[i];
            if (scaleVal > voxelScale) {
                level = i;
                break;
            }
        }
        if (level == 0) {
            upperLayerWeight = 0.0f;
            upperLevel = 0;
        } else if (level == N_LEVELS) {
            upperLayerWeight = 1.0f;
            upperLevel = N_LEVELS - 1;
        } else {
            float coarserScale = cGridScales[level - 1];
            float finerScale = cGridScales[level];
            upperLayerWeight = (coarserScale - scaleVal) / (coarserScale - finerScale);
            upperLevel = level;
        }
    }
    {   // Lower(coarser) level, fit top-down
        int level = -1;
        for (int i = N_LEVELS - 1; i >= 0; --i) {
            float voxelScale = cGridScales[i];
            if (scaleVal < voxelScale) {
                level = i;
                break;
            }
        }
        if (level == -1) {
            lowerLayerWeight = 1.0f;
            lowerLevel = 0;
        } else if (level == N_LEVELS - 1) {
            lowerLayerWeight = 0.0f;
            lowerLevel = N_LEVELS - 1;
        } else {
            float coarserScale = cGridScales[level];
            float finerScale = cGridScales[level + 1];
            lowerLayerWeight = (scaleVal - finerScale) / (coarserScale - finerScale);
            lowerLevel = level;
        }
    }

    const float* upperGrid = grids + cGridOffsets[upperLevel] * N_FEATURES_PER_LEVEL;
    const float* lowerGrid = grids + cGridOffsets[lowerLevel] * N_FEATURES_PER_LEVEL;
    
    uint32_t upperRes = cResolutions[upperLevel];
    uint32_t lowerRes = cResolutions[lowerLevel];
    
    float3 upperPosGrid = make_float3(pos3.x * upperRes, pos3.y * upperRes, pos3.z * upperRes);
    float3 lowerPosGrid = make_float3(pos3.x * lowerRes, pos3.y * lowerRes, pos3.z * lowerRes);
    uint3 upperBase = make_uint3(upperPosGrid.x, upperPosGrid.y, upperPosGrid.z);
    uint3 lowerBase = make_uint3(lowerPosGrid.x, lowerPosGrid.y, lowerPosGrid.z);
    float3 upperOffset = make_float3(upperPosGrid.x - upperBase.x, upperPosGrid.y - upperBase.y, upperPosGrid.z - upperBase.z);
    float3 lowerOffset = make_float3(lowerPosGrid.x - lowerBase.x, lowerPosGrid.y - lowerBase.y, lowerPosGrid.z - lowerBase.z);

    float upperEntry[MAX_FEATURES_PER_LEVEL];
    float lowerEntry[MAX_FEATURES_PER_LEVEL];
    #pragma unroll
    for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) {
        upperEntry[i] = 0.0f;
        lowerEntry[i] = 0.0f;
    };

    #pragma unroll
    for (int corner = 0; corner < 8; corner++) {
        // Get hash index
        uint3 corner3 = _corner_offset(corner);
        uint3 upperIndex3 = make_uint3(upperBase.x + corner3.x, upperBase.y + corner3.y, upperBase.z + corner3.z);
        uint3 lowerIndex3 = make_uint3(lowerBase.x + corner3.x, lowerBase.y + corner3.y, lowerBase.z + corner3.z);
        uint32_t upperIndex = _hash_index<LOG2_HASHMAP_SIZE>(upperIndex3, upperLevel);
        uint32_t lowerIndex = _hash_index<LOG2_HASHMAP_SIZE>(lowerIndex3, lowerLevel);

        // Calculate interpolation weight
        float upperWeight = (corner3.x ? upperOffset.x : (1 - upperOffset.x)) *
                            (corner3.y ? upperOffset.y : (1 - upperOffset.y)) *
                            (corner3.z ? upperOffset.z : (1 - upperOffset.z));
        float lowerWeight = (corner3.x ? lowerOffset.x : (1 - lowerOffset.x)) *
                            (corner3.y ? lowerOffset.y : (1 - lowerOffset.y)) *
                            (corner3.z ? lowerOffset.z : (1 - lowerOffset.z));
        
        const float* upperFeatures = upperGrid + upperIndex * N_FEATURES_PER_LEVEL;
        const float* lowerFeatures = lowerGrid + lowerIndex * N_FEATURES_PER_LEVEL;

        #pragma unroll
        for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) {
            upperEntry[i] += upperFeatures[i] * upperWeight;
            lowerEntry[i] += lowerFeatures[i] * lowerWeight;
        }
    }

    // Write to output
    #pragma unroll
    for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) {
        pixOut[i] = lowerEntry[i] * lowerLayerWeight + upperEntry[i] * upperLayerWeight;
    }
}

template<int N_LEVELS, int LOG2_HASHMAP_SIZE, int N_FEATURES_PER_LEVEL>
__global__ void backwardKernel(
    const float* grids,
    const float* clsInput,
    const float* dL_doutput,
    float* dL_dgrids,
    uint32_t count,
    uint32_t fullDim
) {
    uint32_t pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixIdx >= count) return;
    
    const float* dL_dpixOut = dL_doutput + pixIdx * fullDim;

    const float *pixIn = clsInput + pixIdx * fullDim + N_FEATURES_PER_LEVEL;
    float3 pos3 = make_float3(pixIn[0], pixIn[1], pixIn[2]);
    float scaleVal = pixIn[6];

    // Get corresponding level and layer interpolation weight
    // Lower level means coarser, larger voxel size
    int upperLevel, lowerLevel;
    float upperLayerWeight = 0.0f;
    float lowerLayerWeight = 0.0f;

    {   // Upper(finer) level, fit bottom-up
        int level = N_LEVELS;
        for (int i = 0; i < N_LEVELS; ++i) {
            float voxelScale = cGridScales[i];
            if (scaleVal > voxelScale) {
                level = i;
                break;
            }
        }
        if (level == 0) {
            upperLayerWeight = 0.0f;
            upperLevel = 0;
        } else if (level == N_LEVELS) {
            upperLayerWeight = 1.0f;
            upperLevel = N_LEVELS - 1;
        } else {
            float coarserScale = cGridScales[level - 1];
            float finerScale = cGridScales[level];
            upperLayerWeight = (coarserScale - scaleVal) / (coarserScale - finerScale);
            upperLevel = level;
        }
    }
    {   // Lower(coarser) level, fit top-down
        int level = -1;
        for (int i = N_LEVELS - 1; i >= 0; --i) {
            float voxelScale = cGridScales[i];
            if (scaleVal < voxelScale) {
                level = i;
                break;
            }
        }
        if (level == -1) {
            lowerLayerWeight = 1.0f;
            lowerLevel = 0;
        } else if (level == N_LEVELS - 1) {
            lowerLayerWeight = 0.0f;
            lowerLevel = N_LEVELS - 1;
        } else {
            float coarserScale = cGridScales[level];
            float finerScale = cGridScales[level + 1];
            lowerLayerWeight = (scaleVal - finerScale) / (coarserScale - finerScale);
            lowerLevel = level;
        }
    }

    float* dL_dupperGrid = dL_dgrids + cGridOffsets[upperLevel] * N_FEATURES_PER_LEVEL;
    float* dL_dlowerGrid = dL_dgrids + cGridOffsets[lowerLevel] * N_FEATURES_PER_LEVEL;
    
    uint32_t upperRes = cResolutions[upperLevel];
    uint32_t lowerRes = cResolutions[lowerLevel];
    
    float3 upperPosGrid = make_float3(pos3.x * upperRes, pos3.y * upperRes, pos3.z * upperRes);
    float3 lowerPosGrid = make_float3(pos3.x * lowerRes, pos3.y * lowerRes, pos3.z * lowerRes);
    uint3 upperBase = make_uint3(upperPosGrid.x, upperPosGrid.y, upperPosGrid.z);
    uint3 lowerBase = make_uint3(lowerPosGrid.x, lowerPosGrid.y, lowerPosGrid.z);
    float3 upperOffset = make_float3(upperPosGrid.x - upperBase.x, upperPosGrid.y - upperBase.y, upperPosGrid.z - upperBase.z);
    float3 lowerOffset = make_float3(lowerPosGrid.x - lowerBase.x, lowerPosGrid.y - lowerBase.y, lowerPosGrid.z - lowerBase.z);

    // Write to output
    float dL_dupperEntry[MAX_FEATURES_PER_LEVEL];
    float dL_dlowerEntry[MAX_FEATURES_PER_LEVEL];
    #pragma unroll
    for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) {
        dL_dupperEntry[i] = dL_dpixOut[i] * upperLayerWeight;
        dL_dlowerEntry[i] = dL_dpixOut[i] * lowerLayerWeight;
    }

    #pragma unroll
    for (int corner = 0; corner < 8; corner++) {
        // Get hash index
        uint3 corner3 = _corner_offset(corner);
        uint3 upperIndex3 = make_uint3(upperBase.x + corner3.x, upperBase.y + corner3.y, upperBase.z + corner3.z);
        uint3 lowerIndex3 = make_uint3(lowerBase.x + corner3.x, lowerBase.y + corner3.y, lowerBase.z + corner3.z);
        uint32_t upperIndex = _hash_index<LOG2_HASHMAP_SIZE>(upperIndex3, upperLevel);
        uint32_t lowerIndex = _hash_index<LOG2_HASHMAP_SIZE>(lowerIndex3, lowerLevel);

        // Calculate interpolation weight
        float upperWeight = (corner3.x ? upperOffset.x : (1 - upperOffset.x)) *
                            (corner3.y ? upperOffset.y : (1 - upperOffset.y)) *
                            (corner3.z ? upperOffset.z : (1 - upperOffset.z));
        float lowerWeight = (corner3.x ? lowerOffset.x : (1 - lowerOffset.x)) *
                            (corner3.y ? lowerOffset.y : (1 - lowerOffset.y)) *
                            (corner3.z ? lowerOffset.z : (1 - lowerOffset.z));
        
        float* dL_dupperFeatures = dL_dupperGrid + upperIndex * N_FEATURES_PER_LEVEL;
        float* dL_dlowerFeatures = dL_dlowerGrid + lowerIndex * N_FEATURES_PER_LEVEL;

        #pragma unroll
        for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) {
            atomicAdd(&dL_dupperFeatures[i], dL_dupperEntry[i] * upperWeight);
            atomicAdd(&dL_dlowerFeatures[i], dL_dlowerEntry[i] * lowerWeight);
        }
    }
}


namespace HashGridInterp {
    void initializeConstants(
        uint32_t nLevels,
        uint32_t nFeaturesPerLevel,
        uint32_t log2HashMapSize,
        uint32_t baseResolution,
        float perLevelScale,
        float interpRatio
    ) {
        float resolution = baseResolution;
        uint32_t offset = 0;
        std::vector<uint32_t> resolutions;
        std::vector<uint32_t> gridSizes;
        std::vector<uint32_t> gridOffsets;
        std::vector<float> gridScales;

        for (int i = 0; i < nLevels; i++) {
            uint64_t elems64 = (uint64_t) resolution + 1;
            elems64 = elems64 * elems64 * elems64;
            uint32_t elems = (uint32_t) std::min(elems64, (uint64_t) (1ull << log2HashMapSize));
            float scale = interpRatio / resolution;

            resolutions.push_back((uint32_t) resolution);
            gridSizes.push_back(elems);
            gridOffsets.push_back(offset);
            gridScales.push_back(scale);

            resolution *= perLevelScale;
            offset += elems;
        }

        cudaMemcpyToSymbol(cResolutions, resolutions.data(), nLevels * sizeof(uint32_t));
        cudaMemcpyToSymbol(cGridSizes, gridSizes.data(), nLevels * sizeof(uint32_t));
        cudaMemcpyToSymbol(cGridOffsets, gridOffsets.data(), nLevels * sizeof(uint32_t));
        cudaMemcpyToSymbol(cGridScales, gridScales.data(), nLevels * sizeof(float));

        assert(cudaGetLastError() == cudaSuccess && "Failed to set constants for HashGridInterp");
    }

    void launchForward(
        cudaStream_t stream,
        const float* grids,
        float* output,
        uint32_t count,
        uint32_t fullDim,
        uint32_t nLevels,
        uint32_t nFeaturesPerLevel,
        uint32_t log2HashMapSize,
        uint32_t baseResolution,
        float perLevelScale,
        float interpRatio
    ) {
        dim3 block(256);
        dim3 grid((count + 255) / 256);

        switch (nFeaturesPerLevel) {
            case 8: switch (log2HashMapSize) {
                case 19: switch (nLevels) {
                    case 4: forwardKernel<4, 19, 8><<<grid, block, 0, stream>>>(grids, output, count, fullDim); break;
                    case 8: forwardKernel<8, 19, 8><<<grid, block, 0, stream>>>(grids, output, count, fullDim); break;
                    default: assert(false);
                } break;
                default: assert(false);
            } break;
            default: assert(false);
        }
    }

    void launchBackward(
        cudaStream_t stream,
        const float* grids,
        const float* output,
        const float* dL_doutput,
        float* dL_dgrids,
        uint32_t count,
        uint32_t fullDim,
        uint32_t nLevels,
        uint32_t nFeaturesPerLevel,
        uint32_t log2HashMapSize,
        uint32_t baseResolution,
        float perLevelScale,
        float interpRatio
    ) {
        dim3 block(256);
        dim3 grid((count + 255) / 256);

        switch (nFeaturesPerLevel) {
            case 8: switch (log2HashMapSize) {
                case 19: switch (nLevels) {
                    case 4: backwardKernel<4, 19, 8><<<grid, block, 0, stream>>>(grids, output, dL_doutput, dL_dgrids, count, fullDim); break;
                    case 8: backwardKernel<8, 19, 8><<<grid, block, 0, stream>>>(grids, output, dL_doutput, dL_dgrids, count, fullDim); break;
                    default: assert(false);
                } break;
                default: assert(false);
            } break;
            default: assert(false);
        }
    }
}
