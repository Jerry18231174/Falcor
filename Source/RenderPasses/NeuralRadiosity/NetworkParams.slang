

struct NetworkOutput
{
    float data[16];

    float3 toFloat3()
    {
        return float3(data[0], data[1], data[2]);
    }

    float4 toFloat4()
    {
        return float4(data[0], data[1], data[2], data[3]);
    }

    [mutating]
    void clear()
    {
        for (uint i = 0; i < 16; i++) data[i] = 0.f;
    }

    [mutating]
    void fromFloat3(float3 v)
    {
        data[0] = v.x;
        data[1] = v.y;
        data[2] = v.z;
    }

    [mutating]
    void fromFloat4(float4 v)
    {
        data[0] = v.x;
        data[1] = v.y;
        data[2] = v.z;
        data[3] = v.w;
    }
};