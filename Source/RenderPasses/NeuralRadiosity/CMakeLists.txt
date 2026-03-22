add_plugin(NeuralRadiosity)
set_property(TARGET NeuralRadiosity PROPERTY CUDA_ARCHITECTURES 86)

file(GLOB_RECURSE NR_SOURCES
    "*.cpp"
    "*.h"
    "*.cu"
    "*.slang"
)

target_sources(NeuralRadiosity PRIVATE ${NR_SOURCES})

target_link_libraries(NeuralRadiosity PRIVATE
    Falcor
    tiny-cuda-nn
    tiny-cuda-nn-resources
    fmt
)

target_copy_shaders(NeuralRadiosity RenderPasses/NeuralRadiosity)

target_source_group(NeuralRadiosity "RenderPasses")
