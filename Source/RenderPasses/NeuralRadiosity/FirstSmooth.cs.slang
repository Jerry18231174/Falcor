import Scene.Shading;
import Scene.RaytracingInline;
import PathState;
import LoadShadingData;

// Input channels

// Output channels
RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<float4> gFSThp;
RWTexture2D<uint> gActive;
RWTexture2D<float4> gColor;

// Output G-Buffer channels
RWTexture2D<float4> gPos;
RWTexture2D<float4> gDir;
RWTexture2D<float4> gNormal;
RWTexture2D<float4> gAlbedo;
RWTexture2D<float> gRoughness;


#define is_valid(name) (is_valid_##name != 0)

#if !defined(USE_ALPHA_TEST) || !defined(ADJUST_SHADING_NORMALS) || !defined(USE_ENV_LIGHT) || !defined(RAY_FLAGS)
#error "Not all defines are set!"
#endif

struct FirstSmooth
{
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const bool kAdjustShadingNormals = ADJUST_SHADING_NORMALS;
    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const uint kRayFlags = RAY_FLAGS;
    static const float kEnvMapDepth = 100000000.0f; // Arbitrary big number
    static const int kMaxRecursionDepth = 16;
    static const float kSpecularRoughnessThreshold = 0.5f;
    static const uint kRandomBase = 2654435761u;

    uint2 frameDim = {};
    uint frameCount = 0;

    void generatePath(const uint2 pixel, out PathState path)
    {
        path.setActive();
        path.id = pixel.x | (pixel.y << 12);  // | (sampleIdx << 24);
        path.thp = float3(1.f);

        // Create sample generator.
        path.sg = SampleGenerator(pixel, frameCount * kRandomBase);

        // Unspecified:
        // - path.origin
        // - path.dir
        // - path.setHit(hit);
    }

    /** Generates a new scatter ray given a valid BSDF sample.
        \param[in] bs BSDF sample (assumed to be valid).
        \param[in] sd Shading data.
        \param[in] mi Material instance at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const BSDFSample bs, const ShadingData sd, const IMaterialInstance mi, inout PathState path)
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
            bool isDiffuse = bs.isLobe(LobeType::DiffuseReflection) || roughness > kSpecularRoughnessThreshold;

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

    void execute(const uint2 pixel)
    {
        if (any(pixel >= gFirstSmooth.frameDim))
            return;

        // Compute primary ray.
        Ray ray = gScene.camera.computeRayPinhole(pixel, gFirstSmooth.frameDim);
        
        // Trace rays.
        SceneRayQuery<FirstSmooth::kUseAlphaTest> sceneRayQuery;
        HitInfo hit;
        float3 color = float3(0.f);

        // G-Buffer values.
        float3 pos = float3(0.f);
        float3 dir = float3(0.f);
        float3 normal = float3(0.f);
        float3 albedo = float3(0.f);
        float roughness = 0.f;
        float3 thp = float3(0.f);

        // Initialize path state.
        PathState path = {};
        generatePath(pixel, path);

        for (int i = 0; i < FirstSmooth::kMaxRecursionDepth; i++)
        {
            float rayT = 0;
            hit = sceneRayQuery.traceRay(ray, rayT, FirstSmooth::kRayFlags);

            // Miss: Write background color.
            if (!hit.isValid())
            {
                if (kUseEnvLight)
                {
                    color = gScene.envMap.eval(ray.dir) * path.thp;
                }
                path.terminate();
                break;
            }

            // Load shading data. This is a long latency operation.
            ShadingData sd = loadShadingData(hit, ray.origin, ray.dir);

            // Create material instance.
            let lod = ExplicitLodTextureSampler(0.f);
            let hints = getMaterialInstanceHints(hit, i == 0, kAdjustShadingNormals);
            const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
            const BSDFProperties bsdfProperties = mi.getProperties(sd);

            // Update path state.
            path.setHit(hit);
            path.incrementVertexIndex();
            path.sceneLength += float16_t(rayT);
            path.origin = sd.computeRayOrigin();
            path.dir = ray.dir;

            // Emissive surface: terminate path and write emitted radiance.
            if (any(bsdfProperties.emission > 0.f))
            {
                color = bsdfProperties.emission * path.thp;
                path.terminate();
                break;
            }

            // Backside surface: terminate path.
            if (!sd.mtl.isDoubleSided() && !sd.frontFacing)
            {
                path.terminate();
                break;
            }

            // Break the loop if the hit is not on delta only material
            if (!sd.mtl.isDeltaSpecular())
            {
                pos = path.origin;
                dir = path.dir;
                normal = sd.frame.N;
                roughness = bsdfProperties.roughness;
                albedo = all(bsdfProperties.diffuseReflectionAlbedo < 0.01f) ? 
                    bsdfProperties.specularReflectionAlbedo : bsdfProperties.diffuseReflectionAlbedo;

                thp = path.thp * albedo;
                break;
            }
            // uint lobeTypes = mi.getLobeTypes(sd);
            // if ((lobeTypes & (uint)LobeType::Delta) != 0 && (lobeTypes & (uint)LobeType::NonDelta) == 0)

            // Sample next ray segment.
            BSDFSample result;
            bool valid = mi.sample(sd, path.sg, result);
            if (valid) valid = generateScatterRay(result, sd, mi, path);

            // Break the loop if the BSDF sample is invalid.
            if (!valid) break;

            // Update ray.
            ray = path.getScatterRay();
        }

        // Write outputs
        gColor[pixel] = float4(color, 0.f);
        gVBuffer[pixel] = hit.pack();
        gFSThp[pixel] = float4(thp, 0.f);
        gActive[pixel] = path.isActive();

        // Write G-Buffer
        gPos[pixel] = float4(pos, 1.f);
        gDir[pixel] = float4(dir, 0.f);
        gNormal[pixel] = float4(normal, 0.f);
        gAlbedo[pixel] = float4(albedo, 0.f);
        gRoughness[pixel] = roughness;
    }
};

ParameterBlock<FirstSmooth> gFirstSmooth;


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gFirstSmooth.execute(dispatchThreadId.xy);
}