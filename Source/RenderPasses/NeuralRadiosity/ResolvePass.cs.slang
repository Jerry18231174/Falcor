import Scene.Shading;
import NetworkParams;

// Output channels
RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<uint> gActive;
RWTexture2D<float4> gColor;

// Output G-Buffer channels
RWTexture2D<float4> gPos;
RWTexture2D<float4> gDir;
RWTexture2D<float4> gNormal;
RWTexture2D<float4> gAlbedo;
RWTexture2D<float> gRoughness;

// First smooth buffers
RWStructuredBuffer<uint> diffActive;
RWStructuredBuffer<uint> diffIndex;
RWStructuredBuffer<NetworkOutput> diffColor;

// Specular buffers
RWStructuredBuffer<uint> specActive;
RWStructuredBuffer<uint> specIndex;
RWStructuredBuffer<NetworkOutput> specColor;
RWStructuredBuffer<PackedHitInfo> specVBuffer;

// debug
RWStructuredBuffer<float4> debug;


#ifndef RAY_FLAGS
#define RAY_FLAGS true
#endif

#ifndef NUM_CLUSTERS
#define NUM_CLUSTERS 4
#endif

struct ResolvePass
{
    static const uint kRayFlags = RAY_FLAGS;
    static const uint numClusters = NUM_CLUSTERS;

    uint2 frameDim = {};
    uint frameCount = 0;
    uint diffSize = 0;
    uint specSize = 0;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;

        uint pixIdx = pixel.y * frameDim.x + pixel.x;
        float4 thp = gColor[pixel];

        if (specActive[pixIdx] != 0)
        {
            const uint specIdx = specIndex[pixIdx];
            gColor[pixel] = thp * specColor[specIdx].toFloat4();
            // uint visIdx = specIdx * 4;
            // gPos[pixel] = clusterPos[visIdx];
            // gDir[pixel] = clusterPos[visIdx + 1];
            // gNormal[pixel] = clusterPos[visIdx + 2];
            // gAlbedo[pixel] = clusterPos[visIdx + 3];
            // gNormal[pixel] = float4(clusterScale[specIdx * 4], clusterScale[specIdx * 4 + 1], clusterScale[specIdx * 4 + 3], 1.0f);
            // gAlbedo[pixel] = float4(clusterWeight[specIdx * 4], clusterWeight[specIdx * 4 + 1], clusterWeight[specIdx * 4 + 3], 1.0f);
            // gRoughness[pixel] = clusterScale[visIdx];
        }
        else if (diffActive[pixIdx] != 0)
        {
            const uint diffIdx = diffIndex[pixIdx];
            gColor[pixel] = thp * diffColor[diffIdx].toFloat4();
        }

        // if ((pixIdx > 100000) && (pixIdx < 200000))
        // {
        //     gColor[pixel] = debug[pixIdx - 100000];
        // }
    }
}


ParameterBlock<ResolvePass> gResolvePass;


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gResolvePass.execute(dispatchThreadId.xy);
}
