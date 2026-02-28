/*
   UNiT - Shader Library for ReShade.

   Basic Implementation of the Fast Local Laplacian Filtering
   More about it: "https://hal.science/hal-01063419"
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform int _Mode
<
    ui_type = "combo";
    ui_items = "Detail Enhancer\0Local Tonemap\0";
    ui_label = "LLF Mode";
> = 0;

uniform float _Alpha
<
    ui_label = "Contrast Strength";
    ui_type = "drag";
    ui_min = -1.0; ui_max = 1.0;
> = 0.5;

uniform float _Beta
<
    ui_label = "Contrast Sigma";
    ui_type = "drag";
    ui_min = 0.6; ui_max = 1.5;
> = 1.0;

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

#define LUT_SIZE 256

// 4 cascades groups, each group write 3 levels: small -> medium -> large.
// cascade group 1 (1/2, 1/2, 1/4)
texture2D texCascadeHighS	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = RGB10A2; };
sampler sCascadeHighS		{ Texture = texCascadeHighS; };
texture2D texCascadeHighM	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = RGB10A2; };
sampler sCascadeHighM		{ Texture = texCascadeHighM; };
texture2D texCascadeHighL	{ Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = RGB10A2; };
sampler sCascadeHighL		{ Texture = texCascadeHighL; };

// cascade group 2 (1/4, 1/8, 1/8)
texture2D texCascadeMidS	{ Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = RGB10A2; };
sampler sCascadeMidS		{ Texture = texCascadeMidS; };
texture2D texCascadeMidM	{ Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = RGB10A2; };
sampler sCascadeMidM		{ Texture = texCascadeMidM; };
texture2D texCascadeMidL	{ Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = RGB10A2; };
sampler sCascadeMidL		{ Texture = texCascadeMidL; };

// cascade group 3 (1/16, 1/16, 1/32)
texture2D texCascadeLowS	{ Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = RGB10A2; };
sampler sCascadeLowS		{ Texture = texCascadeLowS; };
texture2D texCascadeLowM	{ Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = RGB10A2; };
sampler sCascadeLowM		{ Texture = texCascadeLowM; };
texture2D texCascadeLowL	{ Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = RGB10A2; };
sampler sCascadeLowL		{ Texture = texCascadeLowL; };

// cascade group 4 (1/32, 1/64, 1/64)
texture2D texCascadeBandS	{ Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = RGB10A2; };
sampler sCascadeBandS		{ Texture = texCascadeBandS; };
texture2D texCascadeBandM	{ Width = BUFFER_WIDTH >> 6; Height = BUFFER_HEIGHT >> 6; Format = RGB10A2; };
sampler sCascadeBandM		{ Texture = texCascadeBandM; };
texture2D texCascadeBandL	{ Width = BUFFER_WIDTH >> 6; Height = BUFFER_HEIGHT >> 6; Format = RGB10A2; };
sampler sCascadeBandL		{ Texture = texCascadeBandL; };

// remapping lut
texture2D texWeightsLUT	    { Width = LUT_SIZE; Height = LUT_SIZE; Format = R16F; };
sampler sWeightsLUT		    { Texture = texWeightsLUT; };

/*=============================================================================
/   Global Helper Functions
/============================================================================*/
float dot3(float3 x)
{
    return dot(x, float3(0.2126729, 0.7151522, 0.072175)); 
}

float3 safesqrt(float3 x)
{
    return sqrt(abs(x)) * sign(x);
}

float3 to_gamma(float3 x)
{
    return safesqrt(x);
}

float3 from_gamma(float3 x)
{
    return x * x * sign(x);
}

float3 to_hdr(float3 x)
{
    return x * rsqrt(1.0 - x * x + rcp(255.0));
}

float3 from_hdr(float3 x)
{
    return x * rsqrt(1.0 + x * x);
}

/*=============================================================================
/   Workspace Helper Functions
/============================================================================*/
// Using lut to avoid recalculating weights math on each draw call. Less cache misses?
void weight_lut(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float o : SV_Target)
{
    o = min(LUT_SIZE - 1, rsqrt(uv.x * 3.0) * 64.0) / (LUT_SIZE - 1);
}

float textureLUT(sampler2D s, float3 x)
{
    return tex2Dlod(s, float4(dot3(x) / 3.0, 0, 0, 0)) * (LUT_SIZE - 1);
}

float3 laplacian(float3 higher, float3 lower)
{
    const float sq_pi = 1.77245385;

    float3 gaussian = lower - higher;
    float3 exposure = lower + higher;

    float sigma = textureLUT(sWeightsLUT, exposure);
    float alpha = safesqrt(_Alpha);

    float3 window, ratio;

	// it's can be baked in the LUT
    if (_Mode)
        window = (1.0 - exposure) * rsqrt(sigma);
    else
        window = dot3((alpha > 0 ? 2 : 1) * to_gamma(gaussian)) * rsqrt(sigma);

    ratio = window + sqrt(sigma) * exp(-sigma * abs(gaussian)) * gaussian;

    return ratio * alpha * sq_pi;
}

float2 uniform_vector(in float u, float mip)
{
    float x = u % 2;
    float y = u / 2;

    if (mip != 0) y = round(y);
    
    return float2(x, y) - 0.5;
}

float3 lpass(sampler2D s, float2 uv, float mip, float resolution)
{
    float sigma = exp2(0.5 * mip) * rsqrt(2.0);
    float radius = ceil(sigma * sqrt(2.0));
    float texel = rcp(resolution * 0.5);

    float4 sum = 0.0;

    for(int i = 0; i < (int)radius; i++)
    {
        float3 tap = 0.0;
        
        float2 offset = uniform_vector((float(i) / radius) + 0.5, mip);
        float2 samdir = BUFFER_PIXEL_SIZE * offset * texel;

        float weight = exp(-0.5 * dot(offset, offset) * (radius * radius) / (sigma * sigma));
		tap += tex2Dlod(s, float4(uv + samdir, 0, 0)).rgb * weight;
		tap += tex2Dlod(s, float4(uv - samdir, 0, 0)).rgb * weight;
        
        sum.xyz += tap;
        sum.w += weight;
    }
    
    return sum.xyz / (sum.w * 2.0);
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
void csc_high_s(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(ReShade::BackBuffer, texcoord, 0, 0.5), 1.0);
}
void csc_high_m(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeHighS, texcoord, 1, 0.5), 1.0);
}
void csc_high_l(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeHighM, texcoord, 2, 0.25), 1.0);
}
void csc_mid_s(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeHighL, texcoord, 3, 0.25), 1.0);
}
void csc_mid_m(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeMidS, texcoord, 4, 0.125), 1.0);
}
void csc_mid_l(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeMidM, texcoord, 5, 0.125), 1.0);
}
void csc_low_s(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeMidL, texcoord, 6, 0.0625), 1.0);
}
void csc_low_m(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeLowS, texcoord, 7, 0.0625), 1.0);
}
void csc_low_l(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeLowM, texcoord, 8, 0.03125), 1.0);
}
void csc_band_s(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeLowL, texcoord, 9, 0.03125), 1.0);
}
void csc_band_m(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeBandS, texcoord, 10, 0.015625), 1.0);
}
void csc_band_l(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeBandM, texcoord, 11, 0.015625), 1.0);
}

void FetchLevel(in float2 uv, out float3 L[13])
{
    L[0] = tex2D(ReShade::BackBuffer, uv).rgb;

    L[1] = tex2D(sCascadeHighS, uv).rgb;
    L[2] = tex2D(sCascadeHighM, uv).rgb;
    L[3] = tex2D(sCascadeHighL, uv).rgb;

    L[4] = tex2D(sCascadeMidS, uv).rgb;
    L[5] = tex2D(sCascadeMidM, uv).rgb;
    L[6] = tex2D(sCascadeMidL, uv).rgb;

    L[7] = tex2D(sCascadeLowS, uv).rgb;
    L[8] = tex2D(sCascadeLowM, uv).rgb;
    L[9] = tex2D(sCascadeLowL, uv).rgb;

    L[10] = tex2D(sCascadeBandS, uv).rgb;
    L[11] = tex2D(sCascadeBandM, uv).rgb;
    L[12] = tex2D(sCascadeBandL, uv).rgb;
}

void main(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float3 output : SV_Target)
{
    float3 L[13];
    float3 G[13];

    FetchLevel(texcoord, L);

    float3 L_o = L[0]; // original

    float sigma = rsqrt(exp2(_Beta));
    float2 o = float2(1.0 - sigma, sigma);

    // cascaded gaussians, low-pass upsampling
    for (int i = 1; i <= 11; i++) 
    {
        L[i+1] = L[i] * o.x + L[i+1] * o.y;
    }

    // the second derivative...
    for (int i = 1; i <= 12; i++) 
    {
        G[i] = laplacian(L[i], L[i-1]);
    }
    
    // cascaded laplacians next, high-pass upsampling
    for (int i = 1; i <= 11; i++) 
    { 
        G[i+1] = o.y * G[i] + o.x * G[i+1];
    }

    float3 G_a = 0;

    // final averaging
    for (int i = 1; i <= 12; i++) 
    {
        G_a += G[i];
    }

    G_a /= 12.0;
    
    // x - original, y - laplacians
    float2 mean;
    mean.x = dot3(L_o);
    mean.y = dot3(G_a);
    
    // linear processing gives a bit better visual
    L_o = from_gamma(L_o);
    L_o = to_hdr(L_o);
    L_o *= (mean.x + mean.y / 0.5) / (mean.x);
    L_o = from_hdr(L_o);
    L_o = to_gamma(L_o);

    output = _Debug ? rsqrt(2.0) * (mean.x + mean.y) / (mean.x) : L_o;
}

/*=============================================================================
/   Technique Definition
/============================================================================*/
technique UNiT_Laplacians < 
ui_label = "UNiT: Laplacians";
ui_tooltip = "									UNiT: Fast-LLF \n\n" "___________________________________________________________________________________________\n\n" "Fast and visual-accurate implementation of the local laplacian filtering image enhancement\n" "that capable in real time, similar to that used in professional photo editors.\n\n" " - Developed by RG2PS - "; >
{
    // <> Group 1
    pass { VertexShader = PostProcessVS; PixelShader = csc_high_s; RenderTarget = texCascadeHighS; }
    pass { VertexShader = PostProcessVS; PixelShader = csc_high_m; RenderTarget = texCascadeHighM; }
    pass { VertexShader = PostProcessVS; PixelShader = csc_high_l; RenderTarget = texCascadeHighL; }
    // <> Group 2
    pass { VertexShader = PostProcessVS; PixelShader = csc_mid_s; RenderTarget = texCascadeMidS; }
    pass { VertexShader = PostProcessVS; PixelShader = csc_mid_m; RenderTarget = texCascadeMidM; }
    pass { VertexShader = PostProcessVS; PixelShader = csc_mid_l; RenderTarget = texCascadeMidL; }
    // <> Group 3
    pass { VertexShader = PostProcessVS; PixelShader = csc_low_s; RenderTarget = texCascadeLowS; }
    pass { VertexShader = PostProcessVS; PixelShader = csc_low_m; RenderTarget = texCascadeLowM; }
    pass { VertexShader = PostProcessVS; PixelShader = csc_low_l; RenderTarget = texCascadeLowL; }
    // <> Group 4
    pass { VertexShader = PostProcessVS; PixelShader = csc_band_s; RenderTarget = texCascadeBandS; }
    pass { VertexShader = PostProcessVS; PixelShader = csc_band_m; RenderTarget = texCascadeBandM; }
    pass { VertexShader = PostProcessVS; PixelShader = csc_band_l; RenderTarget = texCascadeBandL; }
    // <>
    pass { VertexShader = PostProcessVS; PixelShader = weight_lut; RenderTarget = texWeightsLUT; }
    pass { VertexShader = PostProcessVS; PixelShader = main; }
}
