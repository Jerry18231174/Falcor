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
    struct Config {
        uint32_t nClusters = 4;
        uint32_t nLevels = 8;
        uint32_t nFeaturesPerLevel = 8;
        uint32_t log2HashMapSize = 19;
        uint32_t baseResolution = 32;
        float perLevelScale = 2.0f;
        float interpRatio = 0.5f;
    };

    void initializeConstants(const Config& config);

    void launchForward(
        cudaStream_t stream,
        const float* grids,
        float* input,
        uint32_t count,
        uint32_t fullDim,
        uint32_t encOffset,
        uint32_t posOffset,
        uint32_t scaleOffset,
        uint32_t weightOffset,
        const Config& config
    );

    void launchBackward(
        cudaStream_t stream,
        const float* grids,
        const float* input,
        const float* dL_dinput,
        float* dL_dgrids,
        uint32_t count,
        uint32_t fullDim,
        uint32_t encOffset,
        uint32_t posOffset,
        uint32_t scaleOffset,
        uint32_t weightOffset,
        const Config& config
    );
}
