/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Scene/SceneDefines.slangh"
__exported import Scene.Shading;
import Scene.Material.ShadingUtils;
import Rendering.Materials.TexLODHelpers;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import PathState;


/** Compute derivatives at primary triangle hit.
*/
void computeDerivativesAtPrimaryTriangleHit(const TriangleHit hit, uint2 pixel, uint2 frameDim, out float2 ddx, out float2 ddy)
{
    // Get world space position
    VertexData v = gScene.getVertexData(hit);
    float hitT = length(v.posW - gScene.camera.getPosition());
    const float3 nonNormalizedRayDir = gScene.camera.computeNonNormalizedRayDirPinhole(pixel, frameDim);
    float3 rayDir = normalize(nonNormalizedRayDir);

    float3 dDdx, dDdy;
    computeRayDirectionDifferentials(nonNormalizedRayDir, gScene.camera.data.cameraU, gScene.camera.data.cameraV, frameDim, dDdx, dDdy);

    RayDiff rd = RayDiff(float3(0.f), float3(0.f), dDdx, dDdy);
    rd = rd.propagate(v.posW.xyz, rayDir, hitT, v.faceNormalW);

    // Bounce using primary hit geometry.
    const float4x4 worldMat = gScene.getWorldMatrix(hit.instanceID);
    const float3x3 worldInvTransposeMat = gScene.getInverseTransposeWorldMatrix(hit.instanceID);
    const uint3 vertexIndices = gScene.getIndices(hit.instanceID, hit.primitiveIndex);
    StaticVertexData vertices[3] = { gScene.getVertex(vertexIndices[0]), gScene.getVertex(vertexIndices[1]), gScene.getVertex(vertexIndices[2]) };
    float2 dBarydx, dBarydy;
    float3 unnormalizedN, normals[3];

    prepareRayDiffAtHitPoint(v, vertices, hit.getBarycentricWeights(), rayDir, worldMat, worldInvTransposeMat, rd, unnormalizedN, normals, dBarydx, dBarydy, ddx, ddy);
}

/** Setup vertex data based on hit point.
    \param[in] hit Hit information.
    \param[in] rayOrigin Ray origin.
    \param[in] rayDir Normalized ray direction.
    \param[out] materialID Material ID.
    \return VertexData struct.
*/
VertexData loadVertexData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir, out uint materialID)
{
    VertexData v = {};
    materialID = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        StaticVertexData vertices[3];
        v = gScene.getVertexData(triangleHit, vertices);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(curveHit);
        materialID = gScene.getMaterialID(curveHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfGridHit = hit.getSDFGridHit();
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);
        materialID = gScene.getMaterialID(sdfGridHit.instanceID);
    }
#endif
    return v;
}

/** Determine hints to use when creating the material instance.
*/
uint getMaterialInstanceHints(const HitInfo hit, const bool isPrimary, const bool adjustShadingNormals)
{
    uint hints = 0;
    if (hit.getType() == HitType::Triangle || hit.getType() == HitType::DisplacedTriangle)
    {
        if (adjustShadingNormals && !isPrimary) hints |= (uint)MaterialInstanceHints::AdjustShadingNormal;
    }
    return hints;
}

/** Setup ShadingData based on loaded vertex/material attributes for a hit point.
    \param[in] hit Hit information.
    \param[in] rayOrigin Ray origin.
    \param[in] rayDir Normalized ray direction.
    \return ShadingData struct.
*/
ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir)
{
    uint materialID = {};

    VertexData v = loadVertexData(hit, rayOrigin, rayDir, materialID);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);
    return sd;
}

/** Generates a new scatter ray given a valid BSDF sample.
    \param[in] bs BSDF sample (assumed to be valid).
    \param[in] sd Shading data.
    \param[in] mi Material instance at the shading point.
    \param[in,out] path The path state.
    \param[in] specularRoughnessThreshold Specular roughness threshold.
    \return True if a ray was generated, false otherwise.
*/
bool generateScatterRay(
    const BSDFSample bs,
    const ShadingData sd,
    const IMaterialInstance mi,
    inout PathState path,
    float specularRoughnessThreshold
)
{
    const bool isTriangleHit = path.hit.getType() == HitType::Triangle;

    path.dir = bs.wo;
    path.thp *= bs.weight;
    path.pdf = bs.pdf;

    path.clearEventFlags();

    // Handle reflection events.
    if (bs.isLobe(LobeType::Reflection))
    {
        // We classify specular events as diffuse if the roughness is above some threshold.
        float roughness = mi.getProperties(sd).roughness;
        bool isDiffuse = bs.isLobe(LobeType::DiffuseReflection) || roughness > specularRoughnessThreshold;

        if (isDiffuse)
        {
            path.incrementBounces(BounceType::Diffuse);
        }
        else
        {
            path.incrementBounces(BounceType::Specular);
            path.setSpecular();
        }
    }

    // Handle delta events.
    if (bs.isLobe(LobeType::Delta))
    {
        path.setDelta();
    }

    // Handle transmission events.
    if (bs.isLobe(LobeType::Transmission))
    {
        path.incrementBounces(BounceType::Transmission);
        path.setTransmission();

        {
            // Compute ray origin for next ray segment.
            path.origin = sd.computeRayOrigin(false);

            // Update interior list and inside volume flag.
            if (!sd.mtl.isThinSurface())
            {
                uint nestedPriority = sd.mtl.getNestedPriority();
                path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                path.setInsideDielectricVolume(!path.interiorList.isEmpty());
            }
        }
    }

    // Save the normal used for NEE. This is needed for MIS.
    path.normal = sd.getOrientedFaceNormal();

    // Mark the path as valid only if it has a non-zero throughput.
    bool valid = any(path.thp > 0.f);

    return valid;
}


/** Types of samplable lights.
*/
enum class LightType
{
    EnvMap,
    Emissive,
    Analytic
};

/** Describes a light sample.
*/
struct LightSample
{
    float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    float   pdf;        ///< Pdf with respect to solid angle at the shading point.
    float3  origin;     ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    float   distance;   ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
    float3  dir;        ///< Ray direction for visibility evaluation (normalized).
    uint    lightType;  ///< Light type this sample comes from (LightType casted to uint).

    Ray getVisibilityRay() { return Ray(origin, dir, 0.f, distance); }
};

/** Select a light type for sampling.
    \param[out] lightType Selected light type.
    \param[out] pdf Probability for selected type.
    \param[in,out] sg Sample generator.
    \return Return true if selection is valid.
*/
bool selectLightType(
    out uint lightType, out float pdf, inout SampleGenerator sg,
    bool useEmissiveLights, bool useAnalyticLights, bool useEnvLight)
{
    float p[3];
    p[0] = useEnvLight ? 1.f : 0.f;
    p[1] = useEmissiveLights ? 1.f : 0.f;
    p[2] = useAnalyticLights ? 1.f : 0.f;

    // Normalize probabilities. Early out if zero.
    float sum = p[0] + p[1] + p[2];
    if (sum == 0.f) return false;

    float invSum = 1.f / sum;
    p[0] *= invSum;
    p[1] *= invSum;
    p[2] *= invSum;

    float u = sampleNext1D(sg);

    [unroll]
    for (lightType = 0; lightType < 3; ++lightType)
    {
        if (u < p[lightType])
        {
            pdf = p[lightType];
            return true;
        }
        u -= p[lightType];
    }

    lightType = {};
    pdf = {};

    return false;
}

/** Generates a light sample on the emissive geometry.
    \param[in] emissiveSampler Emissive light sampler.
    \param[in] sd Shading frame of the sampling surface interaction.
    \param[in] upperHemisphere True if only upper hemisphere should be considered.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateEmissiveSample(
    const EmissiveLightSampler emissiveSampler,
    const ShadingData sd,
    const bool upperHemisphere,
    inout SampleGenerator sg,
    out LightSample ls)
{
    ls = {}; // Default initialization to avoid divergence at returns.

    TriangleLightSample tls;
    if (!emissiveSampler.sampleLight(sd.posW, sd.getOrientedFaceNormal(), upperHemisphere, sg, tls)) return false;

    // Setup returned sample.
    ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
    ls.pdf = tls.pdf;
    // Offset shading and light position to avoid self-intersection.
    float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
    ls.origin = sd.computeRayOrigin();
    float3 toLight = lightPos - ls.origin;
    ls.distance = length(toLight);
    ls.dir = normalize(toLight);

    return any(ls.Li > 0.f);
}

/** Generates a light sample on the analytic lights.
    \param[in] vertex Path vertex.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateAnalyticLightSample(
    const ShadingData sd,
    inout SampleGenerator sg,
    out LightSample ls)
{
    ls = {}; // Default initialization to avoid divergence at returns.

    uint lightCount = gScene.getLightCount();
    if (lightCount == 0) return false;

    // Sample analytic light source selected uniformly from the light list.
    // TODO: Sample based on estimated contributions as pdf.
    uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

    // Sample local light source.
    AnalyticLightSample lightSample;
    if (!sampleLight(sd.posW, gScene.getLight(lightIndex), sg, lightSample)) return false;

    // Setup returned sample.
    ls.pdf = lightSample.pdf / lightCount;
    ls.Li = lightSample.Li * lightCount;
    // Offset shading position to avoid self-intersection.
    ls.origin = sd.computeRayOrigin();
    // Analytic lights do not currently have a geometric representation in the scene.
    // Do not worry about adjusting the ray length to avoid self-intersections at the light.
    ls.distance = lightSample.distance;
    ls.dir = lightSample.dir;

    return any(ls.Li > 0.f);
}

TriangleLightHit getTriangleLightHit(HitInfo hit, ShadingData sd)
{
    TriangleLightHit lightHit;
    TriangleHit triangleHit = hit.getTriangleHit();
    lightHit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
    lightHit.posW = sd.posW;
    lightHit.normalW = sd.getOrientedFaceNormal();
    return lightHit;
}