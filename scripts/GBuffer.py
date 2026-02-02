from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("GBuffer")
    
    GBuffer = createPass("GBufferRT", {"samplePattern": "Center", "sampleCount": 1})
    g.addPass(GBuffer, "GBuffer")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("GBuffer.normW", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None
