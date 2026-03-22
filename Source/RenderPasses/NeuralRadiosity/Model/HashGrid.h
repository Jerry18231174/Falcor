#pragma once
#include <cstdint>
#include <memory>

#ifndef MAX_LEVELS
#define MAX_LEVELS 8
#endif

#ifndef MAX_FEATURES_PER_LEVEL
#define MAX_FEATURES_PER_LEVEL 16
#endif

#define PRIME_X 1
#define PRIME_Y 19349663
#define PRIME_Z 83492791

#define CONCAT  0
#define MEAN    1
#define INTERP  2

#ifndef LAYER_REDUCE
#define LAYER_REDUCE CONCAT
#endif


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
    );

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
    );
}