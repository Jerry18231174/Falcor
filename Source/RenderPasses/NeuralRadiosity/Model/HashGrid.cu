#include "HashGrid.h"
#include <vector>
#include <cuda_runtime.h>
#include <assert.h>


__constant__ uint32_t cResolutions[MAX_LEVELS];
__constant__ uint32_t cGridSizes[MAX_LEVELS];
__constant__ uint32_t cGridOffsets[MAX_LEVELS];


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
    const float* pos,
    const float* grids,
    float* output,
    uint32_t count
) {
    uint32_t pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixIdx >= count) return;
    
#if (LAYER_REDUCE == CONCAT)
    float* pixOut = output + pixIdx * N_LEVELS * N_FEATURES_PER_LEVEL;
#elif (LAYER_REDUCE == MEAN)
    float* pixOut = output + pixIdx * N_FEATURES_PER_LEVEL;
#endif

    const float *posPtr = pos + pixIdx * 4;
    float3 pos3 = make_float3(posPtr[0], posPtr[1], posPtr[2]);

    float entry[MAX_FEATURES_PER_LEVEL];
    #pragma unroll
    for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) { entry[i] = 0.0f; }

    #pragma unroll
    for (int level = 0; level < N_LEVELS; level++) {
#if (LAYER_REDUCE == CONCAT)
        float* levelOut = pixOut + level * N_FEATURES_PER_LEVEL;
        #pragma unroll
        for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) entry[i] = 0.0f;
#endif
        const float* grid = grids + cGridOffsets[level] * N_FEATURES_PER_LEVEL;
        
        uint32_t res = cResolutions[level];
        
        float3 posGrid = make_float3(pos3.x * res, pos3.y * res, pos3.z * res);
        uint3 base = make_uint3(posGrid.x, posGrid.y, posGrid.z);
        float3 offset = make_float3(posGrid.x - base.x, posGrid.y - base.y, posGrid.z - base.z);

        #pragma unroll
        for (int corner = 0; corner < 8; corner++) {
            // Get hash index
            uint3 corner3 = _corner_offset(corner);
            uint3 index3 = make_uint3(base.x + corner3.x, base.y + corner3.y, base.z + corner3.z);
            uint32_t index = _hash_index<LOG2_HASHMAP_SIZE>(index3, level);

            // Calculate interpolation weight
            float weight = (corner3.x ? offset.x : (1 - offset.x)) *
                           (corner3.y ? offset.y : (1 - offset.y)) *
                           (corner3.z ? offset.z : (1 - offset.z));
            
            const float* features = grid + index * N_FEATURES_PER_LEVEL;
            #pragma unroll
            for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) {
                entry[i] += features[i] * weight;
            }
        }

        // Write to output
#if (LAYER_REDUCE == CONCAT)
        #pragma unroll
        for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) { levelOut[i] = entry[i]; }
#endif
    }
#if (LAYER_REDUCE == MEAN)
    #pragma unroll
    for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) { pixOut[i] = entry[i] / N_LEVELS; }
#endif
}

template<int N_LEVELS, int LOG2_HASHMAP_SIZE, int N_FEATURES_PER_LEVEL>
__global__ void backwardKernel(
    const float* pos,
    const float* grids,
    const float* output,
    const float* dL_doutput,
    float* dL_dgrids,
    uint32_t count
) {
    uint32_t pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixIdx >= count) return;
}


__host__ void setConstants(
    uint32_t nLevels,
    uint32_t nFeaturesPerLevel,
    uint32_t log2HashMapSize,
    uint32_t baseResolution,
    float perLevelScale
) {
    float resolution = baseResolution;
    uint32_t offset = 0;
    std::vector<uint32_t> resolutions;
    std::vector<uint32_t> gridSizes;
    std::vector<uint32_t> gridOffsets;

    for (int i = 0; i < nLevels; i++) {
        uint64_t elems64 = (uint64_t) resolution + 1;
        elems64 = elems64 * elems64 * elems64;
        uint32_t elems = (uint32_t) std::min(elems64, (uint64_t) (1ull << log2HashMapSize));

        resolutions.push_back((uint32_t) resolution);
        gridSizes.push_back(elems);
        gridOffsets.push_back(offset);

        resolution *= perLevelScale;
        offset += elems;
    }

    cudaMemcpyToSymbol(cResolutions, resolutions.data(), nLevels * sizeof(uint32_t));
    cudaMemcpyToSymbol(cGridSizes, gridSizes.data(), nLevels * sizeof(uint32_t));
    cudaMemcpyToSymbol(cGridOffsets, gridOffsets.data(), nLevels * sizeof(uint32_t));
}


namespace HashGrid {
    void launchForward(
        const float* pos,
        const float* grids,
        float* output,
        uint32_t count,
        uint32_t nLevels,
        uint32_t nFeaturesPerLevel,
        uint32_t log2HashMapSize,
        uint32_t baseResolution,
        float perLevelScale
    ) {
        dim3 block(256);
        dim3 grid((count + 255) / 256);

        setConstants(nLevels, nFeaturesPerLevel, log2HashMapSize, baseResolution, perLevelScale);

        switch (nFeaturesPerLevel) {
            case 8: switch (log2HashMapSize) {
                case 19: switch (nLevels) {
                    case 4: forwardKernel<4, 19, 8><<<grid, block>>>(pos, grids, output, count); break;
                    case 8: forwardKernel<8, 19, 8><<<grid, block>>>(pos, grids, output, count); break;
                    default: assert(false);
                } break;
                default: assert(false);
            } break;
            default: assert(false);
        }

        // forwardKernel<4, 19, 8><<<grid, block>>>(
        //     pos, grids,
        //     output, count,
        //     nFeaturesPerLevel, baseResolution, perLevelScale
        // );
    }

    void launchBackward(
        const float* pos,
        const float* grids,
        const float* output,
        const float* dL_doutput,
        float* dL_dgrids,
        uint32_t count,
        uint32_t nLevels,
        uint32_t nFeaturesPerLevel,
        uint32_t log2HashMapSize,
        uint32_t baseResolution,
        float perLevelScale
    ) {
        dim3 block(256);
        dim3 grid((count + 255) / 256);

        backwardKernel<4, 19, 8><<<grid, block>>>(
            pos, grids,
            output, dL_doutput,
            dL_dgrids,
            count
        );
    }
}