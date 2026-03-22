#pragma once
#include <cstdint>
#include <memory>
#include <cuda_runtime.h>

#ifndef MAX_LEVELS
#define MAX_LEVELS 8
#endif

#ifndef MAX_FEATURES_PER_LEVEL
#define MAX_FEATURES_PER_LEVEL 16
#endif

#define PRIME_X 1
#define PRIME_Y 19349663
#define PRIME_Z 83492791


namespace HashGridInterp {
    void initializeConstants(
        uint32_t nLevels,
        uint32_t nFeaturesPerLevel,
        uint32_t log2HashMapSize,
        uint32_t baseResolution,
        float perLevelScale,
        float interpRatio
    );

    void launchForward(
        cudaStream_t stream,
        const float* grids,
        float* clsInput,
        uint32_t count,
        uint32_t fullDim,
        uint32_t nLevels,
        uint32_t nFeaturesPerLevel,
        uint32_t log2HashMapSize,
        uint32_t baseResolution,
        float perLevelScale,
        float interpRatio
    );

    void launchBackward(
        cudaStream_t stream,
        const float* grids,
        const float* clsInput,
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
    );
}
