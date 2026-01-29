/*
   UNiT - Shader Library for ReShade.
   Adaptive Frequency Domain Image Detail Restoration via Spectral Diffusion
   Source: https://en.wikipedia.org/wiki/Discrete_cosine_transform

   Sharpen in freq domain is really effective detail enhancement technique because by 
   working with the full-scaled image we pure aproxximate continuous gaussian integral. 
   Its allows us to cover the entire image and inseparably process each pixel.
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform float _Amount
<
    ui_label = "Sharpen Strength";
    ui_type = "slider";
    ui_min = 0.001; ui_max = 1.0;
> = 1.0f;

uniform bool _Debug
<
    ui_label = "Visialize Laplacian";
    ui_type = "radio";
> = false;

/*=============================================================================
/   Buffer Samplers Definition
/============================================================================*/
#include "ReShade.fxh"

#define PI  3.1415926535897932384626433832795

#define fl(x) sqrt(x)
#define tl(x) sign(x) * ((x)*(x))

texture texChannelColor : COLOR;

#ifndef DCT_TILE_SIZE
 #define DCT_TILE_SIZE 8
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
    
    // frequency domain approximation of the laplacian eigenvalue
    // souce: https://www.shadertoy.com/view/MctyD8
    float lx = sin(PI * float(local_x) / (2.0 * float(dct.tile_size)));
    lx = 4.0 * lx * lx;
    
    float ly = sin(PI * float(local_y) / (2.0 * float(dct.tile_size)));
    ly = 4.0 * ly * ly; 
    
    // using the radial deviation to better frequency estimation and reduce shitty tiling
    float fx = float(local_x) / float(dct.tile_size - 1);
    float fy = float(local_y) / float(dct.tile_size - 1);
    float r_xy = sqrt(fx * fx + fy * fy);
    
    return ((lx + ly) - r_xy) * r_xy * sqrt(2.0); 
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
    float weight = exp(-0.125 * lambda);

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

// shader entry points start. spatial > II > freq > III > spatial
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

void main(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float3 output : SV_Target)
{
    // now standard laplacian transform in spatial domain..
    float3 color = tex2Dfetch(sChannelColor, vpos.xy, 0).rgb;
    float3 color_diffuse = max(0, tex2Dfetch(sDCT_III_V, vpos.xy, 0).rgb);
    
    // for laplacian transform better to use the gamma space
    color = fl(color);
    color_diffuse = fl(color_diffuse);

    const float x = 4.0 * PI;
    const float xr = rsqrt(0.5 * x * x);

    float3 delta = color_diffuse - color;
    float3 amplitude = min(x, rsqrt(abs(delta))) * cos(1.0 - delta * PI);
	float3 color_delta = delta * amplitude; 

	color_delta = clamp(color_delta, -xr, xr);
	
    output = _Debug ? dot(tl(color_delta), 1) * 6.0 : tl(color - color_delta * sqrt(_Amount));
}

/*=============================================================================
/   Technique Definition
/============================================================================*/
technique UNiT_Sharpen < 
ui_label = "UNiT: Sharpen";
ui_tooltip = "					    	   UNiT: Frequency-Domain Sharpening \n\n" "________________________________________________________________________________________________\n\n" "A new approach to image sharpening working with full-scaled image instead fixed spatial kernels.\n" "This tech allows the effective restore certain details lost during TAAU/DLSS processing.\n\n" " - Developed by RG2PS - "; >
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