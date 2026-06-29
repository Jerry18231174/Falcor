import Scene.Shading;
import NetworkParams;

// Output channels
RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<uint> gActive;
RWTexture2D<float4> gColor;

Texture2D<float4> tempColor;

// Output G-Buffer channels
RWTexture2D<float4> gPos;
RWTexture2D<float4> gDir;
RWTexture2D<float4> gNormal;
RWTexture2D<float4> gAlbedo;
RWTexture2D<float> gRoughness;

// First smooth buffers
RWStructuredBuffer<uint> diffActive;
RWStructuredBuffer<uint> diffIndex;

// Specular buffers
RWStructuredBuffer<uint> specActive;
RWStructuredBuffer<uint> specIndex;


struct PostFilterPass
{
    uint2 frameDim = {};
    uint frameCount = 0;

    // Parameters
    int kernelRadius;
    float sigmaSpatial;
    float sigmaNormal;
    float sigmaAlbedo;
    float sigmaColor;

    float gaussianWeight(float distance2, float sigma)
    {
        sigma = max(sigma, 1e-4f);
        return exp(-distance2 / (2.0f * sigma * sigma));
    }

    void execute(const uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;
        const uint pixIdx = pixel.y * frameDim.x + pixel.x;

        const float4 centerSample = tempColor[pixel];
        const float3 centerColor = centerSample.xyz;

        if (specActive[pixIdx] != 0)
        {
            const float3 centerNormal = normalize(gNormal[pixel].xyz);
            const float3 centerAlbedo = gAlbedo[pixel].xyz;

            float3 filteredColor = float3(0.f);
            float weightSum = 0.f;

            const int radius = kernelRadius;

            for (int dy = -radius; dy <= radius; ++dy)
            {
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    const int2 samplePixelI = int2(pixel) + int2(dx, dy);
                    // if (any(samplePixelI < 0) || any(samplePixelI >= int2(frameDim))) continue;

                    const uint2 samplePixel = uint2(samplePixelI);
                    const uint samplePixIdx = samplePixel.y * frameDim.x + samplePixel.x;

                    // if (specActive[samplePixIdx] == 0 || gActive[samplePixel] == 0) continue;

                    const float3 sampleColor = tempColor[samplePixel].xyz;
                    const float3 sampleNormal = normalize(gNormal[samplePixel].xyz);
                    const float3 sampleAlbedo = gAlbedo[samplePixel].xyz;

                    const float spatialDist2 = float(dx * dx + dy * dy);
                    const float normalDist2 = 1.f - saturate(dot(centerNormal, sampleNormal));
                    const float albedoDist2 = dot(centerAlbedo - sampleAlbedo, centerAlbedo - sampleAlbedo);
                    const float colorDist2 = dot(centerColor - sampleColor, centerColor - sampleColor);

                    const float weight =
                        gaussianWeight(spatialDist2, sigmaSpatial) *
                        gaussianWeight(normalDist2, sigmaNormal) *
                        gaussianWeight(albedoDist2, sigmaAlbedo) *
                        gaussianWeight(colorDist2, sigmaColor);

                    filteredColor += weight * sampleColor;
                    weightSum += weight;
                }
            }

            gColor[pixel] = float4(weightSum > 0.f ? filteredColor / weightSum : centerColor, centerSample.w);
            return;
        }

        gColor[pixel] = centerSample;
    }
}


ParameterBlock<PostFilterPass> gPostFilterPass;


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gPostFilterPass.execute(dispatchThreadId.xy);
}
