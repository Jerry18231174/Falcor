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
RWStructuredBuffer<DiffNetInput> diffInput;

// Specular buffers
RWStructuredBuffer<uint> specActive;
RWStructuredBuffer<uint> specIndex;
RWStructuredBuffer<SpecNetInput> specInput;
RWStructuredBuffer<PackedHitInfo> specVBuffer;
RWStructuredBuffer<uint2> specPixel;


struct CompactPass
{
    uint2 frameDim = {};
    uint frameCount = 0;
    uint diffSize = 0;
    uint specSize = 0;

    float3 bboxMin;
    float3 bboxMax;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;
        const uint pixIdx = pixel.y * frameDim.x + pixel.x;

        if (diffActive[pixIdx] != 0)
        {
            const uint diffIdx = diffIndex[pixIdx];
            writeDiffInputGBuffer(
                diffInput, diffIdx,
                (gPos[pixel].xyz - bboxMin) / (bboxMax - bboxMin),
                gDir[pixel].xyz,
                gNormal[pixel].xyz,
                gAlbedo[pixel].xyz,
                gRoughness[pixel]
            );
        }

        if (specActive[pixIdx] != 0)
        {
            const uint specIdx = specIndex[pixIdx];
            specVBuffer[specIdx] = gVBuffer[pixel];
            specPixel[specIdx] = pixel;
            writeSpecInputGBuffer(
                specInput, specIdx,
                gPos[pixel].xyz,
                gDir[pixel].xyz,
                gNormal[pixel].xyz,
                gAlbedo[pixel].xyz,
                gRoughness[pixel]
            );
        }
    }
}


ParameterBlock<CompactPass> gCompactPass;


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gCompactPass.execute(dispatchThreadId.xy);
}
