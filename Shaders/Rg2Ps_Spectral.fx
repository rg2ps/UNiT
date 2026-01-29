/*
   UNiT - Shader Library for ReShade.
   Chromatic Aberration via Visibility Spectrum Integration.

   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform float _Curvature
<
    ui_label = "Curvature";
    ui_type = "slider";
    ui_min = -1.0; ui_max = 1.0;
> = 0.5;

uniform float _Amount
<
    ui_label = "Strength";
    ui_type = "slider";
    ui_min = -1.0; ui_max = 1.0;
> = -0.25;

uniform bool _GamutMap
<
    ui_type = "radio";
    ui_label = "DCI-P3 CA";
    ui_tooltip = "Using a wider color gamut to achieve a more vibrant effect";
> = false;

/*=============================================================================
/   Buffer Samplers Definition
/============================================================================*/
#include "ReShade.fxh"

#ifndef MAX_SEGMENTS
    #define MAX_SEGMENTS 24
#endif 

texture texHDRChannel
{
    Width = BUFFER_WIDTH; 
    Height = BUFFER_HEIGHT; 
    Format = RGBA16F;
};

sampler sHDRChannel
{ 
    Texture = texHDRChannel; 
};

texture texCMFLUT < source ="cmf_lut.png" ; >
{
    Width = 1024; 
    Height = 1; 
    Format = RGBA8;
};

sampler sCMFLUT
{ 
    Texture = texCMFLUT; 
};

texture texPrismMarch
{
    Width = BUFFER_WIDTH; 
    Height = BUFFER_HEIGHT; 
    Format = RGBA16F;
};

sampler sPrismMarch
{ 
    Texture = texPrismMarch; 
};

/*=============================================================================
/   Global Helper Functions
/============================================================================*/
float3 from_linear(float3 x)
{
    return sqrt(x);
}

float3 to_linear(float3 x)
{
    return x * x;
}

// aces curve and inverse of it
float3 from_hdr(float3 x)
{
    return saturate((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f));
}

float3 to_hdr(float3 x)
{
   return max((-0.59f * x + 0.03f - sqrt(-1.0127f * x*x + 1.3702f * x + 0.0009f)) / (2.0f * (2.43f*x - 2.51f)), 1e-6f);
}

float3 to_dcip3(float3 x)
{
    float3x3 m = float3x3
    (
        0.822461969,  0.033194199,  0.017082631,
        0.177538031,  0.966805801, -0.009801631,
        0.000000000,  0.000000000,  0.992718000
    );

    return _GamutMap ? mul(x, m) : x;
}

float3 from_dcip3(float3 x)
{
    float3x3 m = float3x3
    (
        1.224940063, -0.042056965, -0.019637167,
       -0.224940063,  1.042056965,  0.019637167,
        0.000000000,  0.000000000,  1.007320000
    );

    return _GamutMap ? mul(x, m) : x;
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
void write_hdr(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    output = float4(to_hdr(to_linear(tex2D(ReShade::BackBuffer, texcoord).rgb)), 1.0);
}

void get_optical_distance(inout float lambda, in float r)
{
    // forward and inner angle of incidence of a ray on the lens
    float a1 = atan(r); // [0 - π/2]
    float a2 = 1.570796326 - a1; // [π/2 - 0]
    float theta = lerp(a2, a1, _Curvature * 0.5 + 0.5);   
    lambda = theta * (lambda - 0.4) * 0.15;
}

void ray_tap(out float2 direction, in float2 coord, in float lambda, in float T)
{
    float2 uv = coord * 2.0 - 1.0;
    float r = length(uv);

    get_optical_distance(lambda, r);

    float step_length = _Amount * T;
    float2 ray_tap = uv * lambda * step_length;

    direction = (uv - ray_tap) * 0.5 + 0.5;
}

int get_samplecount()
{
    float M = 0.125; // The LUT requires at least 8 segments per pixel
    float v = M + sqrt(abs(_Amount)) * (rcp(MAX_SEGMENTS) - M);
    return (int)ceil(1/v);
}

void prism_integral(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    float3 sum = 0.0;
    float3 total = 0.0;
	int _sample = get_samplecount();

    float step = 400.0 / float(_sample);

    [loop]
    for (int i = 0; i < _sample; i++)
    {
        float lambda = (float(i) + 0.5f) / float(_sample);

	    float2 pos;
        ray_tap(pos, texcoord, lambda, 1);
	
	    float3 color = to_dcip3(tex2Dlod(sHDRChannel, float4(pos, 0, 0)).rgb);
        float3 weight = tex2Dfetch(sCMFLUT, int2(lambda * 1023.0, 0), 0).rgb * step;

	    sum += color * weight;
	    total += weight;
    }

    const int _refiement = 8;

    float2 local_div[_refiement];

    for (int i = 0; i < _refiement; i++)
    {
        float lambda = (float(i) + 0.5f) / float(_refiement);
        float2 pos;
        ray_tap(pos, texcoord, lambda, 1);
        local_div[i] = pos;
    }

    float2 mean_div = 0;

    [unroll]
    for (int i = 0; i < _refiement; i++)
    {
        mean_div += local_div[i];
    }

    mean_div /= float(_refiement);
    
    float divergence = 0;

    for (int i = 0; i < _refiement; i++)
    {
        divergence = max(divergence, length(local_div[i] - mean_div));
    }
    
    output = float4(sum / total, sqrt(divergence));
}

void main(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float3 output : SV_Target0)
{
    float3 sum = 0.0;
    float total = 0.0;

    float step = 44.44;

    float divergence = tex2Dlod(sPrismMarch, float4(texcoord, 0, 0)).a;

    [unroll]
    for(int i = 0; i < 9; i++)
    {
        float lambda = (float(i) + 0.5f) / 9.0;

	    float2 pos;
        ray_tap(pos, texcoord, lambda, divergence);

	    float3 color = tex2Dlod(sPrismMarch, float4(pos, 0, 0)).rgb;
        
        sum += color * step;
        total += step;
    }

    output = from_linear(from_dcip3(from_hdr(sum / total)));
}

/*=============================================================================
/   Technique Definition
/============================================================================*/
technique UNiTSpectral < 
ui_label = "UNiT: Spectral";
ui_tooltip = "                      UNiT: Spectral Chromatic Aberration \n\n" "_______________________________________________________________________________\n\n" "Physically based approach of chromatic aberration effect that uses a spectrum\n" "integration to simulate the realistic prism dispersion visual.\n\n" " - Developed by RG2PS - "; > 
{
    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = write_hdr;
	    RenderTarget = texHDRChannel;
    }

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = prism_integral;
	    RenderTarget = texPrismMarch;
    }

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = main;
    }
}
