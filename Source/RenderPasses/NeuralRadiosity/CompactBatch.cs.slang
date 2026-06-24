import Scene.Shading;
import NetworkParams;

// Ray batch buffers
RWStructuredBuffer<float4> allPos;
RWStructuredBuffer<float4> allDir;
RWStructuredBuffer<float4> allNormal;
RWStructuredBuffer<float4> allAlbedo;
RWStructuredBuffer<float> allRoughness;
RWStructuredBuffer<PackedHitInfo> allVBuffer;
RWStructuredBuffer<float4> allColor;
RWStructuredBuffer<float4> allEmission;

RWStructuredBuffer<uint> diffActive;
RWStructuredBuffer<uint> diffIndex;
RWStructuredBuffer<DiffNetInput> diffInput;
RWStructuredBuffer<NetworkOutput> diffColor;

RWStructuredBuffer<uint> specActive;
RWStructuredBuffer<uint> specIndex;
RWStructuredBuffer<SpecNetInput> specInput;
RWStructuredBuffer<NetworkOutput> specColor;
RWStructuredBuffer<PackedHitInfo> specVBuffer;
RWStructuredBuffer<uint2> specPixel;


struct CompactBatch
{
    uint batchSize = 0;
    uint batchCount = 0;
    uint diffSize = 0;
    uint specSize = 0;

    float3 bboxMin;
    float3 bboxMax;

    void execute(const uint idx)
    {
        if (any(idx >= batchSize)) return;

        if (diffActive[idx] != 0)
        {
            const uint diffIdx = diffIndex[idx];
            writeDiffInputGBuffer(
                diffInput, diffIdx,
                (allPos[idx].xyz - bboxMin) / (bboxMax - bboxMin),
                allDir[idx].xyz,
                allNormal[idx].xyz,
                allAlbedo[idx].xyz,
                allRoughness[idx]
            );
        }

        if (specActive[idx] != 0)
        {
            const uint specIdx = specIndex[idx];
            specVBuffer[specIdx] = allVBuffer[idx];
            specPixel[specIdx] = uint2(idx & 0xFFFu, idx >> 12);
            writeSpecInputGBuffer(
                specInput, specIdx,
                allPos[idx].xyz,
                allDir[idx].xyz,
                allNormal[idx].xyz,
                allAlbedo[idx].xyz,
                allRoughness[idx]
            );
        }
    }
}


ParameterBlock<CompactBatch> gCompactBatch;


[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gCompactBatch.execute(dispatchThreadId.x);
}
