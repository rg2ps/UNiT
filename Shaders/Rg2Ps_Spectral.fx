/*
   UNiT - Shader Library for ReShade.
   Chromatic Aberration via Visibility Spectrum Integration.

   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform float S
<
    ui_label = "Curvature";
    ui_type = "slider";
    ui_min = -1.0; ui_max = 1.0;
> = 0.5;

uniform float CA
<
    ui_label = "Strength";
    ui_type = "slider";
    ui_min = -1.0; ui_max = 1.0;
> = 0.25;

/*=============================================================================
/   Buffer Samplers Definition
/============================================================================*/
#include "ReShade.fxh"

#ifndef MAX_SEGMENTS
    #define MAX_SEGMENTS 24
#endif 

// Hardcoded, for better don't touch it
#define LUT_SIZE 1024
#define LUT_SEGMENTS 8

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
    Width = LUT_SIZE; 
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
float3 safesqrt(float3 x)
{
    return sqrt(abs(x));
}

float3 from_linear(float3 x)
{
    return safesqrt(x) * sign(x);
}

float3 to_linear(float3 x)
{
    return x * x;
}

float3 from_hdr(float3 x)
{
    return x * rsqrt(1.0 + x * x);
}

float3 to_hdr(float3 x)
{
   return x * rsqrt(1.0 - x * x + (1.0 / 255.0));
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
void write_hdr(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    float3 sdr = tex2D(ReShade::BackBuffer, texcoord).rgb;
    output = float4(to_hdr(to_linear(sdr)), 1.0);
}

void get_optical_distance(inout float lambda, in float r)
{
    // forward and inner angle of incidence of a ray on the lens
    float a1 = atan(r); // [0 - π/2]
    float a2 = 1.570796326 - a1; // [π/2 - 0]
    float theta = lerp(a2, a1, S * 0.5 + 0.5);   
    lambda = theta * (lambda - 0.4) * 0.15;
}

void ray_tap(out float2 direction, in float2 coord, in float lambda, in float mult)
{
    float2 uv = coord * 2.0 - 1.0;
    float radius = length(uv);

    get_optical_distance(lambda, radius);

    float step = CA * mult;
    float2 tap = uv * lambda * step;

    direction = (uv + tap) * 0.5 + 0.5;
}

// the lut requires at least 8 color segments per pixel
int get_samplecount()
{
    float a = rcp(LUT_SEGMENTS), b = rcp(MAX_SEGMENTS);
    return (int)ceil(rcp(a + (b - a) * safesqrt(CA)));
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
	    float2 pos; ray_tap(pos, texcoord, lambda, 1);
	
	    float3 color = tex2Dlod(sHDRChannel, float4(pos, 0, 0)).rgb;
        float3 weight = tex2Dfetch(sCMFLUT, int2(lambda * (LUT_SIZE - 1), 0), 0).rgb * step;

	    sum += color * weight;
	    total += weight;
    }

    float2 local_div[LUT_SEGMENTS];
    float2 mean_div = 0;

    for (int i = 0; i < LUT_SEGMENTS; i++)
    {
        float lambda = (float(i) + 0.5f) / float(LUT_SEGMENTS);
        ray_tap(local_div[i], texcoord, lambda, 1);
        mean_div += local_div[i];
    }

    mean_div /= float(LUT_SEGMENTS);

    float divergence = 0;

    for (int i = 0; i < LUT_SEGMENTS; i++)
    {
        divergence = max(divergence, length(local_div[i] - mean_div));
    }
    
    output = float4(from_hdr(sum / total), sqrt(divergence));
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
	    float2 pos; ray_tap(pos, texcoord, lambda, divergence);

	    float3 color = tex2Dlod(sPrismMarch, float4(pos, 0, 0)).rgb;
        
        sum += color * step;
        total += step;
    }

    output = from_linear(sum / total);
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
