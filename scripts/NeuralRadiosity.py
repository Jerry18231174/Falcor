from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("NeuralRadiosity")
    
    NeuralRadiosity = createPass("NeuralRadiosity")
    g.addPass(NeuralRadiosity, "NeuralRadiosity")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    OptixDenoiser = createPass("OptixDenoiser", {})
    g.addPass(OptixDenoiser, "Denoiser")

    # g.addEdge("NeuralRadiosity.color", "ToneMapper.src")
    g.addEdge("NeuralRadiosity.color", "Denoiser.color")
    g.addEdge("Denoiser.output", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")

    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None
