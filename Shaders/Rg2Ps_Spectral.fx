/*
   UNiT - Shader Library for ReShade.
   Chromatic Aberration via Visibility Spectrum Integration.

   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform float RAY_DISPERSION
<
    ui_label = "Curvature";
    ui_type = "slider";
    ui_min = -1.0; ui_max = 1.0;
> = 0.5;

uniform float RAY_LENGTH
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

#ifndef LUT_SIZE
	#define LUT_SIZE 1024
#endif 

#ifndef MIN_COLOR_POINTS
	#define MIN_COLOR_POINTS 8
#endif 

#define PI 3.14159265358979323
#define H_PI (PI / 2.0)

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

texture texCMFLUT_RGB < source ="cmf_lut.png" ; >
{
    Width = LUT_SIZE; 
    Height = 1; 
    Format = RGBA8;
};

sampler sCMFLUT_RGB
{ 
    Texture = texCMFLUT_RGB; 
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
float3 from_hdr(float3 x)
{
    return sqrt(1.0 - exp2(-x));
}

float3 to_hdr(float3 x)
{
   return -log2(1.0 - min(0.999, x * x));
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
void hdrc(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    output = float4(to_hdr(tex2D(ReShade::BackBuffer, texcoord).rgb), 1.0);
}

float optical_distance(in float lambda, in float r)
{
    // forward and inner angle of incidence of a ray on the lens
    float angle = atan(r); // [0 - π/2]
    float inverseangle = H_PI - angle; // [π/2 - 0]
    float theta = lerp(inverseangle, angle, RAY_DISPERSION * 0.5 + 0.5);   
    return theta * (lambda - 0.4) / (PI * 1.4142);
}

void ray_tap(out float2 direction, in float2 uv, in float lambda, in float k)
{
    float2 coord = uv * 2.0 - 1.0;
    float radius = sqrt(dot(coord, coord));

    float depth = optical_distance(lambda, radius);
    float2 distortion = coord * depth * RAY_LENGTH;

    direction = (coord + distortion * k) * 0.5 + 0.5;
}

void prism(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    float3 sum = 0.0;
    float3 total = 0.0;

    float2 divpos[MIN_COLOR_POINTS];
    float2 avgpos = 0;
    
    float divergence = 0;

    for (int i = 0; i < MIN_COLOR_POINTS; i++)
    {
        float lambda = (float(i) + 0.5f) / float(MIN_COLOR_POINTS);
        
        ray_tap(divpos[i], texcoord, lambda, 1.0);
        
        avgpos += divpos[i];
    }

    avgpos /= float(MIN_COLOR_POINTS);

    for (int i = 0; i < MIN_COLOR_POINTS; i++)
    {
        divergence = max(divergence, sqrt(dot(divpos[i] - avgpos, divpos[i] - avgpos)));
    }

    float div = sqrt(divergence);
	
	int M = ceil(MIN_COLOR_POINTS + abs(RAY_LENGTH) * exp(-divergence) * MAX_SEGMENTS);

    float step = 400.0 / float(M);

    [loop]
    for (int i = 0; i < M; i++)
    {
        float lambda = (float(i) + 0.5f) / float(M);        
	    float2 pos; ray_tap(pos, texcoord, lambda, 1.0);
	
	    float3 color = tex2Dlod(sHDRChannel, float4(pos, 0, 0)).rgb;
        float3 weight = tex2Dfetch(sCMFLUT_RGB, int2(lambda * (LUT_SIZE - 1), 0), 0).xyz * step;

	    sum += color * weight;
	    total += weight;
    }
    
    output = float4(from_hdr(sum / total), div);
}

void main(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float3 output : SV_Target0)
{
    float3 sum = 0.0;
    float total = 0.0;

    float step = 44.44;

    float divergence = tex2Dlod(sPrismMarch, float4(texcoord, 0, 0)).a;

    for(int i = 0; i < 9; i++)
    {
        float lambda = (float(i) + 0.5f) / 9.0;
	    float2 pos; ray_tap(pos, texcoord, lambda, divergence);

	    float3 color = tex2Dlod(sPrismMarch, float4(pos, 0, 0)).rgb;
        
        sum += color * step;
        total += step;
    }

    output = sum / total;
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
	    PixelShader = hdrc;
	    RenderTarget = texHDRChannel;
    }

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = prism;
	    RenderTarget = texPrismMarch;
    }

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = main;
    }
}
