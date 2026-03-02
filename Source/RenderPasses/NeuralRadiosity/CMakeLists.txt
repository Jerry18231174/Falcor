add_plugin(NeuralRadiosity)

target_sources(NeuralRadiosity PRIVATE
    NeuralRadiosity.cpp
    NeuralRadiosity.h
    FirstSmooth.cs.slang
    LoadShadingData.slang
    PathState.slang
)

target_copy_shaders(NeuralRadiosity RenderPasses/NeuralRadiosity)

target_source_group(NeuralRadiosity "RenderPasses")
