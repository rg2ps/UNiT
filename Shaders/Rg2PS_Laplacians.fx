/*
   UNiT - Shader Library for ReShade.
   Real-Time Implementation of the Fast Local Laplacian Filtering
   More about it: "https://hal.science/hal-01063419"
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform float _Alpha
<
    ui_label = "Contrast Strength";
    ui_type = "drag";
    ui_min = -1.0; ui_max = 1.0;
> = 0.5;

uniform bool _Debug
<
    ui_label = "Visualize Laplacians";
    ui_type = "radio";
    ui_category_closed = true;
> = false;

/*=============================================================================
/   Buffer Samplers Definition
/============================================================================*/
#include "ReShade.fxh"

#define TARGET_LPL 5 // Lod Per Level

texture2D texLaplacianWeightLUT	    { Width = 256; Height = 256; Format = RG16F; };
texture2D texGaussianCascadeHigh_0	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = RGB10A2; MipLevels = TARGET_LPL; };
texture2D texGaussianCascadeHigh_1	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = RGB10A2; MipLevels = TARGET_LPL; };
texture2D texGaussianCascadeHigh_2	{ Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = RGB10A2; MipLevels = TARGET_LPL; };
texture2D texGaussianCascadeMidl_0	{ Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = RGB10A2; MipLevels = TARGET_LPL; };
texture2D texGaussianCascadeMidl_1	{ Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = RGB10A2; MipLevels = TARGET_LPL; };
texture2D texGaussianCascadeMidl_2	{ Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = RGB10A2; MipLevels = TARGET_LPL; };
texture2D texGaussianCascadeLarg_0	{ Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = RGB10A2; MipLevels = TARGET_LPL; };
texture2D texGaussianCascadeLarg_1	{ Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = RGB10A2; MipLevels = TARGET_LPL; };
texture2D texGaussianCascadeLarg_2	{ Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = RGB10A2; MipLevels = TARGET_LPL; };
texture2D texGaussianCascadeBand_0	{ Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = RGB10A2; MipLevels = TARGET_LPL; };
texture2D texGaussianCascadeBand_1	{ Width = BUFFER_WIDTH >> 6; Height = BUFFER_HEIGHT >> 6; Format = RGB10A2; MipLevels = TARGET_LPL; };
texture2D texGaussianCascadeBand_2	{ Width = BUFFER_WIDTH >> 6; Height = BUFFER_HEIGHT >> 6; Format = RGB10A2; MipLevels = TARGET_LPL; };
sampler sLaplacianWeightLUT		    { Texture = texLaplacianWeightLUT;    MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR; };
sampler sGaussianCascadeHigh_0		{ Texture = texGaussianCascadeHigh_0; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR; };
sampler sGaussianCascadeHigh_1		{ Texture = texGaussianCascadeHigh_1; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR; };
sampler sGaussianCascadeHigh_2		{ Texture = texGaussianCascadeHigh_2; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR; };
sampler sGaussianCascadeMidl_0		{ Texture = texGaussianCascadeMidl_0; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR; };
sampler sGaussianCascadeMidl_1		{ Texture = texGaussianCascadeMidl_1; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR; };
sampler sGaussianCascadeMidl_2		{ Texture = texGaussianCascadeMidl_2; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR; };
sampler sGaussianCascadeLarg_0		{ Texture = texGaussianCascadeLarg_0; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR; };
sampler sGaussianCascadeLarg_1		{ Texture = texGaussianCascadeLarg_1; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR; };
sampler sGaussianCascadeLarg_2		{ Texture = texGaussianCascadeLarg_2; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR; };
sampler sGaussianCascadeBand_0		{ Texture = texGaussianCascadeBand_0; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR; };
sampler sGaussianCascadeBand_1		{ Texture = texGaussianCascadeBand_1; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR; };
sampler sGaussianCascadeBand_2		{ Texture = texGaussianCascadeBand_2; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR; };

/*=============================================================================
/   Workspace Helper Functions
/============================================================================*/
float dot3(float3 x)
{
    return (x.r + x.g + x.b) / 3.0;
}

float safe_sqrt(float x)
{
    return sqrt(abs(x)) * sign(x);
}

float3 to_hdr(float3 x)
{
    return x * rsqrt(2.0 - x * x); 
}

float3 from_hdr(float3 x)
{
    return x * rsqrt(0.5 + 0.5 * x * x);
}

// Using lut to avoid recalculating weights math on each draw call.
// Less cache misses?
void weight_lut(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float2 o : SV_Target)
{
    const float x = 255.0;
    float g = (uv.x * 2.0) - 1.0;
    float i = uv.y * 3.0;
    o.x = min(x, 63.0 * rsqrt(i)) / x;
    o.y = saturate(1.0 - sqrt(abs(g)) * sign(g));
}

float3 remap_laplacian(float3 current, float3 previous)
{
    float3 g = previous - current;
    float3 i = previous + current;

    float2 uv = float2(dot3(g) * 0.5 + 0.5, dot3(i) / 3.0);
    float2 lambda = tex2Dlod(sLaplacianWeightLUT, float4(uv, 0, 0)).rg;

    float sigma = lambda.r * 255.0; 
    float weight = lambda.g * 1.18920711;
    
    float3 ratio = sqrt(5.22485 * sigma) * exp(-abs(g) * sigma) * g;
    
    return (g + ratio) * weight * safe_sqrt(_Alpha);
}

float erf(float x)
{
    // https://www.academia.edu/9730974
    float a = 0.140012;
    float x2 = x * x;
    float k = (1.27323954473 + x2 * a) / (1.0 + x2 * a);
    float t = exp(-x2 * k);
    return sign(x) * sqrt(1.0 - t);
}

float cdf_1d(float sigma, float t0, float t1, float t) 
{
    float s_sqrt2 = sigma * 1.41421356237;

    float a = erf((t - t0) / s_sqrt2);
    float b = erf((t - t1) / s_sqrt2);
    
    return (a - b) * 0.5;
}

float3 t2d(sampler2D s, float2 uv, float L, float scale)
{
    float sigma_pixels = 0.7071 * exp2(L * 0.5);
    float2 texel_size = BUFFER_PIXEL_SIZE * rsqrt(scale);
    
    float sigma_uv = sigma_pixels * texel_size.x;
    float2 cell_size_uv = texel_size;
    
    float omega_texel = (4.0 * 3.1415926) / (6.0 * rcp(scale) * rcp(scale));
    float sigma_mip = clamp(0.5 * log2(rcp(ceil(sigma_pixels)) / omega_texel), 0, TARGET_LPL - 1);

    float radius = ceil(sigma_pixels * exp2(-0.7071 * sigma_mip));

    float3 sum = 0.0;
    float total = 0.0;
    
    for(int i = -radius; i <= radius; i++)
    for(int j = -radius; j <= radius; j++)
    {
        float2 cell_center = float2(i, j) * cell_size_uv;
        
        float2 cell_min = cell_center - cell_size_uv * 0.5;
        float2 cell_max = cell_center + cell_size_uv * 0.5;
        
        float weight_x = cdf_1d(sigma_uv, cell_min.x, cell_max.x, 0);
        float weight_y = cdf_1d(sigma_uv, cell_min.y, cell_max.y, 0);
        float weight = weight_x * weight_y;
        
        float2 position = uv + cell_center;
        
        float3 color = tex2Dlod(s, float4(position, 0, sigma_mip)).rgb;
        
        sum += color * weight;
        total += weight;
    }
    
    return sum / max(total, 1e-6);
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
void csc_high_0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = float4(t2d(ReShade::BackBuffer, texcoord, 0, 0.5), 1.0);
}

void csc_high_1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = float4(t2d(sGaussianCascadeHigh_0, texcoord, 1, 0.5), 1.0);
}

void csc_high_2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = float4(t2d(sGaussianCascadeHigh_1, texcoord, 2, 0.25), 1.0);
}

void csc_midl_0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = float4(t2d(sGaussianCascadeHigh_2, texcoord, 3, 0.25), 1.0);
}

void csc_midl_1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = float4(t2d(sGaussianCascadeMidl_0, texcoord, 4, 0.125), 1.0);
}

void csc_midl_2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = float4(t2d(sGaussianCascadeMidl_1, texcoord, 5, 0.125), 1.0);
}

void csc_larg_0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = float4(t2d(sGaussianCascadeMidl_2, texcoord, 6, 0.0625), 1.0);
}

void csc_larg_1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = float4(t2d(sGaussianCascadeLarg_0, texcoord, 7, 0.0625), 1.0);
}

void csc_larg_2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = float4(t2d(sGaussianCascadeLarg_1, texcoord, 8, 0.03125), 1.0);
}

void csc_band_0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = float4(t2d(sGaussianCascadeLarg_2, texcoord, 9, 0.03125), 1.0);
}

void csc_band_1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = float4(t2d(sGaussianCascadeBand_0, texcoord, 10, 0.015625), 1.0);
}

void csc_band_2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = float4(t2d(sGaussianCascadeBand_1, texcoord, 11, 0.015625), 1.0);
}

void main(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float3 output : SV_Target0)
{
    float3 L_0 = tex2Dfetch(ReShade::BackBuffer, vpos.xy, 0).rgb; // original

    float3 L[13];
    float3 G[13];

    L[0]  = L_0;
    L[1]  = tex2D(sGaussianCascadeHigh_0, texcoord).rgb;
    L[2]  = tex2D(sGaussianCascadeHigh_1, texcoord).rgb;
    L[3]  = tex2D(sGaussianCascadeHigh_2, texcoord).rgb;
    L[4]  = tex2D(sGaussianCascadeMidl_0, texcoord).rgb;
    L[5]  = tex2D(sGaussianCascadeMidl_1, texcoord).rgb;
    L[6]  = tex2D(sGaussianCascadeMidl_2, texcoord).rgb;
    L[7]  = tex2D(sGaussianCascadeLarg_0, texcoord).rgb;
    L[8]  = tex2D(sGaussianCascadeLarg_1, texcoord).rgb;
    L[9]  = tex2D(sGaussianCascadeLarg_2, texcoord).rgb;
    L[10] = tex2D(sGaussianCascadeBand_0, texcoord).rgb;
    L[11] = tex2D(sGaussianCascadeBand_1, texcoord).rgb;
    L[12] = tex2D(sGaussianCascadeBand_2, texcoord).rgb;

    float2 o = float2(1.0 - rsqrt(2.0), rsqrt(2.0));

    // cascaded gaussians, something like low-pass upsampling
    for (int i = 1; i <= 11; i++) 
    {
        L[i+1] = L[i] * o.x + L[i+1] * o.y;
    }

    // aproxximate the second derivative...
    for (int i = 1; i <= 12; i++) 
    {
        G[i] = remap_laplacian(L[i], L[i-1]);
    }
    
    // cascaded laplacians next, high-pass upsampling
    for (int i = 1; i <= 11; i++) 
    { 
        G[i+1] = o.y * G[i] + o.x * G[i+1];
    }

    float3 G_a = 0;

    for (int i = 1; i <= 12; i++) 
    {
        G_a += G[i];
    }

    G_a /= 12.0;

    L_0 = to_hdr(L_0);

    // x - original, y - laplacians
    float2 mean;
    mean.x = _Alpha > 0 ? dot3(L_0) : sqrt(dot3(L_0));
    mean.y = dot3(G_a);
    L_0 *= (mean.x + mean.y) / (mean.x + 1e-6);

    L_0 = from_hdr(L_0);

    output = _Debug ? mean.y * 8.0 : L_0;
}

/*=============================================================================
/   Technique Definition
/============================================================================*/
technique UNiT_Laplacians < 
ui_label = "UNiT: Laplacians";
ui_tooltip = "									UNiT: Fast-LLF \n\n" "________________________________________________________________________________________\n\n" "Fast and visual-accurate implementation of the local laplacian filtering based contrast\n" "enhancement for real time usage, similar to that used in professional photo editors.\n\n" " - Developed by RG2PS - "; >
{
    #define CASCADE_HIGH(i) pass { VertexShader = PostProcessVS; PixelShader = csc_high_##i; RenderTarget = texGaussianCascadeHigh_##i; }
    #define CASCADE_MIDL(i) pass { VertexShader = PostProcessVS; PixelShader = csc_midl_##i; RenderTarget = texGaussianCascadeMidl_##i; }
    #define CASCADE_LARG(i) pass { VertexShader = PostProcessVS; PixelShader = csc_larg_##i; RenderTarget = texGaussianCascadeLarg_##i; }
    #define CASCADE_BAND(i) pass { VertexShader = PostProcessVS; PixelShader = csc_band_##i; RenderTarget = texGaussianCascadeBand_##i; }

    CASCADE_HIGH(0)
    CASCADE_HIGH(1)
    CASCADE_HIGH(2)
    CASCADE_MIDL(0)
    CASCADE_MIDL(1)
    CASCADE_MIDL(2)
    CASCADE_LARG(0)
    CASCADE_LARG(1)
    CASCADE_LARG(2)
    CASCADE_BAND(0)
    CASCADE_BAND(1)
    CASCADE_BAND(2)

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = weight_lut;
        RenderTarget = texLaplacianWeightLUT;
    }

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = main;
    }
}
