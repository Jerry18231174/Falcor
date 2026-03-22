import Scene.Shading;

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
RWStructuredBuffer<float4> diffPos;
RWStructuredBuffer<float4> diffDir;
RWStructuredBuffer<float4> diffNormal;
RWStructuredBuffer<float4> diffAlbedo;
RWStructuredBuffer<float> diffRoughness;
RWStructuredBuffer<uint> diffActive;
RWStructuredBuffer<uint> diffIndex;

// Specular buffers
RWStructuredBuffer<float4> specPos;
RWStructuredBuffer<float4> specDir;
RWStructuredBuffer<float4> specNormal;
RWStructuredBuffer<float4> specAlbedo;
RWStructuredBuffer<float> specRoughness;
RWStructuredBuffer<PackedHitInfo> specVBuffer;
RWStructuredBuffer<uint> specActive;
RWStructuredBuffer<uint> specIndex;


struct CompactPass
{
    uint2 frameDim = {};
    uint frameCount = 0;
    uint diffSize = 0;
    uint specSize = 0;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;
        const uint pixIdx = pixel.y * frameDim.x + pixel.x;

        if (diffActive[pixIdx] != 0)
        {
            const uint diffIdx = diffIndex[pixIdx];
            diffPos[diffIdx] = gPos[pixel];
            diffDir[diffIdx] = gDir[pixel];
            diffNormal[diffIdx] = gNormal[pixel];
            diffAlbedo[diffIdx] = gAlbedo[pixel];
            diffRoughness[diffIdx] = gRoughness[pixel];
        }

        if (specActive[pixIdx] != 0)
        {
            const uint specIdx = specIndex[pixIdx];
            specPos[specIdx] = gPos[pixel];
            specDir[specIdx] = gDir[pixel];
            specNormal[specIdx] = gNormal[pixel];
            specAlbedo[specIdx] = gAlbedo[pixel];
            specRoughness[specIdx] = gRoughness[pixel];
            specVBuffer[specIdx] = gVBuffer[pixel];
        }
    }
}


ParameterBlock<CompactPass> gCompactPass;


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gCompactPass.execute(dispatchThreadId.xy);
}
