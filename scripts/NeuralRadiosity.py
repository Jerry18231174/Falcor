from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("FirstSmoothDebug")
    
    NeuralRadiosity = createPass("NeuralRadiosity")
    g.addPass(NeuralRadiosity, "NeuralRadiosity")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("NeuralRadiosity.fsThp", "ToneMapper.src")

    g.markOutput("NeuralRadiosity.fsThp")
    g.markOutput("NeuralRadiosity.normal")
    g.markOutput("NeuralRadiosity.albedo")
    g.markOutput("ToneMapper.dst")

    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None
