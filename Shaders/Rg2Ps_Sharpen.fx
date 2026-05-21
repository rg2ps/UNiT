/*
   UNiT - Shader Library for ReShade.
   Image Sharpening via Frequency Domain Convolution
   Source: https://en.wikipedia.org/wiki/Discrete_cosine_transform

   Sharpening in frequency domain is really effective detail enhancement 
   technique because by working with the full-scaled image we cover the entire 
   image and inseparably process each pixel.
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform float STRENGTH
<
    ui_label = "Sharpening Strength";
    ui_type = "slider";
    ui_min = 0.001; ui_max = 1.0;
> = 1.0f;

uniform bool DEBUG
<
    ui_label = "Visialize Sharpening";
    ui_type = "radio";
> = false;

/*=============================================================================
/   Buffer Samplers Definition
/============================================================================*/
#include "ReShade.fxh"

#define PI 3.1415926535897932384626433832795

texture texChannelColor : COLOR;

#ifndef DCT_TILE_SIZE
    #define DCT_TILE_SIZE 8
#endif 

#ifndef SIGMA_CONVOLUTION
    #define SIGMA_CONVOLUTION 2.0
#endif 

sampler sChannelColor
{ 
    Texture = texChannelColor;  
    SRGBTexture = true; 
};

texture texDCT_II_H
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA16F;
};

sampler sDCT_II_H
{ 
    Texture = texDCT_II_H; 
};

texture texDCT_II_V
{

    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA16F;
};

sampler sDCT_II_V
{ 
    Texture = texDCT_II_V; 
};

texture texDCT_III_H
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA16F;
};

sampler sDCT_III_H
{ 
    Texture = texDCT_III_H; 
};

texture texDCT_III_V
{

    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA16F;
};

sampler sDCT_III_V
{ 
    Texture = texDCT_III_V; 
};

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
struct block
{
    int tile_size;
    int2 grid_size;
};

block DCT()
{
    block dct;
    dct.tile_size = DCT_TILE_SIZE;
    dct.grid_size = int2(BUFFER_SCREEN_SIZE) / dct.tile_size;
    return dct;
}

float l_eigenvalue(float2 xy, in block dct)
{
    int local_x = int(xy.x) % dct.tile_size;
    int local_y = int(xy.y) % dct.tile_size;

    float fx = float(local_x) / float(dct.tile_size - 1);

    float lx = sin(PI * float(local_x) / (2.0 * float(dct.tile_size)));
    lx = 4.0 * lx * lx;
    
    float ly = sin(PI * float(local_y) / (2.0 * float(dct.tile_size)));
    ly = 4.0 * ly * ly; 

    float fy = float(local_y) / float(dct.tile_size - 1);

    float radial = sqrt(fx * fx + fy * fy) * 0.7071;
    
    return (lx + ly + radial) * radial; 
}

float3 dct_IIe_row(sampler2D s, float2 xy)
{
    block dct = DCT(); 

    int tiles_x = dct.grid_size.x;
    int tile_x = int(xy.x) / dct.tile_size;
    int local_x = int(xy.x) % dct.tile_size;
    int j = int(xy.y);

    int actual_tile_size = dct.tile_size;
    
    if (local_x >= actual_tile_size) return 0;

    float o = PI * float(local_x) / float(actual_tile_size);
    float3 a = 0;
    
    for (int k = 0; k < actual_tile_size; ++k)
    {
        int global_x = tile_x * dct.tile_size + k;
        float3 s = tex2Dfetch(s, int2(global_x, j), 0).xyz;
        float t = (float(k) + 0.5) * o;
        a = a + cos(t) * s;
    }
    
    if (local_x == 0)
        a = a * rsqrt(float(actual_tile_size));
    else
        a = a * sqrt(2.0 / float(actual_tile_size));
    
    return a;
}

float3 dct_IIe_col(sampler2D s, float2 xy)
{
    block dct = DCT(); 

    float lambda = l_eigenvalue(xy, dct);
    float weight = exp(-lambda / (SIGMA_CONVOLUTION * SIGMA_CONVOLUTION));

    int tiles_y = dct.grid_size.y;
    int tile_y = int(xy.y) / dct.tile_size;
    int local_y = int(xy.y) % dct.tile_size;
    int i = int(xy.x);

    int actual_tile_size = dct.tile_size;
    
    if (local_y >= actual_tile_size) return 0;

    float o = PI * float(local_y) / float(actual_tile_size);
    float3 a = 0;
    
    for (int k = 0; k < actual_tile_size; ++k)
    {
        int global_y = tile_y * dct.tile_size + k;
        float3 s = tex2Dfetch(s, int2(i, global_y), 0).xyz;
        float t = (float(k) + 0.5) * o;
        a = a + cos(t) * s;
    }
    
    if (local_y == 0)
        a = a * rsqrt(float(actual_tile_size));
    else
        a = a * sqrt(2.0 / float(actual_tile_size));
    
    return a * weight;
}

float3 dct_IIIe_row(sampler2D s, float2 xy)
{
    block dct = DCT(); 

    int tiles_x = dct.grid_size.x;
    int tile_x = int(xy.x) / dct.tile_size;
    int local_x = int(xy.x) % dct.tile_size;
    int j = int(xy.y);

    int actual_tile_size = dct.tile_size;
    
    if (local_x >= actual_tile_size) return 0;

    float o = PI * (float(local_x) + 0.5) / float(actual_tile_size);
    float3 a = 0;
    
    for (int k = 0; k < actual_tile_size; ++k)
    {
        int global_x = tile_x * dct.tile_size + k;
        float3 s = tex2Dfetch(s, int2(global_x, j), 0).xyz;
        float t = float(k) * o;
        a = a + cos(t) * s;
        
        if (k == 0)
            a = a * rsqrt(2.0);
    }
    
    a = a * sqrt(2.0 / float(actual_tile_size));

    return a;
}

float3 dct_IIIe_col(sampler2D s, float2 xy)
{
    block dct = DCT(); 

    int tiles_y = dct.grid_size.y;
    int tile_y = int(xy.y) / dct.tile_size;
    int local_y = int(xy.y) % dct.tile_size;
    int i = int(xy.x);

    int actual_tile_size = dct.tile_size;
    
    if (local_y >= actual_tile_size) return 0;

    float factor = PI * (float(local_y) + 0.5) / float(actual_tile_size);
    float3 a = 0;
    
    for (int k = 0; k < actual_tile_size; ++k)
    {
        int global_y = tile_y * dct.tile_size + k;
        float3 s = tex2Dfetch(s, int2(i, global_y), 0).xyz;
        float t = float(k) * factor;
        a = a + cos(t) * s;
        
        if (k == 0)
            a = a * rsqrt(2.0);
    }
    
    a = a * sqrt(2.0 / float(actual_tile_size));

    return a;
}

/*=============================================================================
/   Shader entry points start: spatial -> II -> frequency -> III -> spatial
/============================================================================*/
void dct_II_H(float4 vpos : SV_Position, out float4 output : SV_Target)
{
    output = float4(dct_IIe_row(sChannelColor, vpos.xy), 1.0);
}

void dct_II_V(float4 vpos : SV_Position, out float4 output : SV_Target)
{
    output = float4(dct_IIe_col(sDCT_II_H, vpos.xy), 1.0);
}

void dct_III_H(float4 vpos : SV_Position, out float4 output : SV_Target)
{
    output = float4(dct_IIIe_row(sDCT_II_V, vpos.xy), 1.0);
}

void dct_III_V(float4 vpos : SV_Position, out float4 output : SV_Target)
{
    output = float4(dct_IIIe_col(sDCT_III_H, vpos.xy), 1.0);
}

float3 do_remap(in float3 x, out float k)
{
	const float HALF_PI = 1.57079;
	
	float sigma = sqrt(2.0);
	float sigma_sqr = sigma * sigma;

    // For each DCT decomposition level, the extent to which the Laplacian (the difference between adjacent levels) 
    // fits within the linear range of the s-curve [-π/(2σ²), π/(2σ²)] is calculated. 
    // If the Laplacian falls outside this range, the s-curve saturates, and k → 1—this indicates a boundary case where the DCT reconstruction is unreliable. 
    // At such points, the signal reverts to the original, preventing halo artifacts.
    float3 responce = sin(clamp(x * sigma_sqr, -HALF_PI, HALF_PI)) / sigma_sqr;
    float error = saturate(dot(responce, responce));

    k = error;
	
	return x * sqrt(SIGMA_CONVOLUTION) * 2.0;
}

void main(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float3 output : SV_Target)
{
    float3 p0 = tex2Dfetch(sChannelColor, vpos.xy, 0).rgb;     // signal
    float3 p1 = tex2Dfetch(sDCT_II_H, vpos.xy, 0).rgb;         // II-H
    float3 p2 = tex2Dfetch(sDCT_II_V, vpos.xy, 0).rgb;         // II-V  
    float3 p3 = tex2Dfetch(sDCT_III_H, vpos.xy, 0).rgb;        // III-H
    float3 p4 = tex2Dfetch(sDCT_III_V, vpos.xy, 0).rgb;        // III-V (diffusion)

    float3 L1 = p0 - p1;
    float3 L2 = p1 - p2;
    float3 L3 = p2 - p3; 
    float3 L4 = p3 - p4; 

    float3 guide = p0; // p4
    
    float3 signal = DEBUG ? 0.0 : guide;

    float k_max = 0;
    float k;

    signal += do_remap(L4, k); k_max = max(k_max, k);
    signal += do_remap(L3, k); k_max = max(k_max, k);
    signal += do_remap(L2, k); k_max = max(k_max, k);
    signal += do_remap(L1, k); k_max = max(k_max, k);

    signal = DEBUG ? lerp(sqrt(dot(signal, signal) / 3.0), 0, k) : lerp(signal, p0, k);
    
    output = signal;
}

/*=============================================================================
/   Technique Definition
/============================================================================*/
technique UNiT_Sharpen < 
ui_label = "UNiT: Sharpen";
ui_tooltip = "					    	   UNiT: Frequency Domain Sharpening \n\n" "________________________________________________________________________________________________\n\n" "A new approach to image sharpening working with full-scaled image instead fixed spatial kernels.\n" "This tech allows the effective restore certain details lost during TAAU/DLSS processing.\n\n" " - Developed by RG2PS - "; >
{
    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = dct_II_H;
	    RenderTarget = texDCT_II_H;
    }
    
    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = dct_II_V;
	    RenderTarget = texDCT_II_V;
    }
    
    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = dct_III_H;
	    RenderTarget = texDCT_III_H;
    }
    
    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = dct_III_V;
	    RenderTarget = texDCT_III_V;
    }
    
    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = main;
	    SRGBWriteEnable = true;
    }
}
