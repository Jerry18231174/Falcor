import Scene.Shading;
import NetworkParams;

// Ray batch buffers
RWStructuredBuffer<float4> allPos;
RWStructuredBuffer<float4> allDir;
RWStructuredBuffer<float4> allNormal;
RWStructuredBuffer<float4> allAlbedo;
RWStructuredBuffer<float> allRoughness;
RWStructuredBuffer<PackedHitInfo> allVBuffer;
RWStructuredBuffer<NetworkOutput> allColor;

RWStructuredBuffer<float4> diffPos;
RWStructuredBuffer<float4> diffDir;
RWStructuredBuffer<float4> diffNormal;
RWStructuredBuffer<float4> diffAlbedo;
RWStructuredBuffer<float> diffRoughness;
RWStructuredBuffer<uint> diffActive;
RWStructuredBuffer<uint> diffIndex;
RWStructuredBuffer<NetworkOutput> diffColor;

RWStructuredBuffer<float4> specPos;
RWStructuredBuffer<float4> specDir;
RWStructuredBuffer<float4> specNormal;
RWStructuredBuffer<float4> specAlbedo;
RWStructuredBuffer<float> specRoughness;
RWStructuredBuffer<PackedHitInfo> specVBuffer;
RWStructuredBuffer<uint> specActive;
RWStructuredBuffer<uint> specIndex;
RWStructuredBuffer<NetworkOutput> specColor;

RWStructuredBuffer<float4> clusterPos;
RWStructuredBuffer<float4> clusterDir;
RWStructuredBuffer<float> clusterScale;
RWStructuredBuffer<float> clusterWeight;


struct CompactBatch
{
    uint batchSize = 0;
    uint batchCount = 0;
    uint diffSize = 0;
    uint specSize = 0;

    void execute(const uint idx)
    {
        if (any(idx >= batchSize)) return;

        if (diffActive[idx] != 0)
        {
            const uint diffIdx = diffIndex[idx];
            diffPos[diffIdx] = allPos[idx];
            diffDir[diffIdx] = allDir[idx];
            diffNormal[diffIdx] = allNormal[idx];
            diffAlbedo[diffIdx] = allAlbedo[idx];
            diffRoughness[diffIdx] = allRoughness[idx];
        }

        if (specActive[idx] != 0)
        {
            const uint specIdx = specIndex[idx];
            specPos[specIdx] = allPos[idx];
            specDir[specIdx] = allDir[idx];
            specNormal[specIdx] = allNormal[idx];
            specAlbedo[specIdx] = allAlbedo[idx];
            specRoughness[specIdx] = allRoughness[idx];
            specVBuffer[specIdx] = allVBuffer[idx];
        }
    }
}


ParameterBlock<CompactBatch> gCompactBatch;


[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gCompactBatch.execute(dispatchThreadId.x);
}