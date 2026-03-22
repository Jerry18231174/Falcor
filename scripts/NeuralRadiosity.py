from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("FirstSmoothDebug")
    
    NeuralRadiosity = createPass("NeuralRadiosity")
    g.addPass(NeuralRadiosity, "NeuralRadiosity")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("NeuralRadiosity.color", "ToneMapper.src")

    g.markOutput("NeuralRadiosity.color")
    g.markOutput("ToneMapper.dst")

    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None
