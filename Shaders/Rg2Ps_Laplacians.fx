/*
   UNiT - Shader Library for ReShade.

   Basic Implementation of the Fast Local Laplacian Filtering
   More about it: "https://hal.science/hal-01063419"
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform int _M
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

/*=============================================================================
/   Buffer Samplers Definition
/============================================================================*/
#include "ReShade.fxh"

#define LUT_SIZE 256

// group 1 (1/2, 1/2)
texture2D texCascadeGroup1S	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = RGB10A2; };
sampler sCascadeGroup1S		{ Texture = texCascadeGroup1S; };
texture2D texCascadeGroup1L	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = RGB10A2; };
sampler sCascadeGroup1L		{ Texture = texCascadeGroup1L; };
// group 2 (1/4, 1/4)
texture2D texCascadeGroup2S	{ Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = RGB10A2; };
sampler sCascadeGroup2S		{ Texture = texCascadeGroup2S; };
texture2D texCascadeGroup2L	{ Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = RGB10A2; };
sampler sCascadeGroup2L		{ Texture = texCascadeGroup2L; };
// group 2 (1/8, 1/8)
texture2D texCascadeGroup3S	{ Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = RGB10A2; };
sampler sCascadeGroup3S		{ Texture = texCascadeGroup3S; };
texture2D texCascadeGroup3L	{ Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = RGB10A2; };
sampler sCascadeGroup3L		{ Texture = texCascadeGroup3L; };
// group 4 (1/16, 1/16)
texture2D texCascadeGroup4S	{ Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = RGB10A2; };
sampler sCascadeGroup4S		{ Texture = texCascadeGroup4S; };
texture2D texCascadeGroup4L	{ Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = RGB10A2; };
sampler sCascadeGroup4L		{ Texture = texCascadeGroup4L; };
// group 5  (1/32, 1/32)
texture2D texCascadeGroup5S	{ Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = RGB10A2; };
sampler sCascadeGroup5S		{ Texture = texCascadeGroup5S; };
texture2D texCascadeGroup5L	{ Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = RGB10A2; };
sampler sCascadeGroup5L		{ Texture = texCascadeGroup5L; };
// group 6 (1/64, 1/64)
texture2D texCascadeGroup6S	{ Width = BUFFER_WIDTH >> 6; Height = BUFFER_HEIGHT >> 6; Format = RGB10A2; };
sampler sCascadeGroup6S		{ Texture = texCascadeGroup6S; };
texture2D texCascadeGroup6L	{ Width = BUFFER_WIDTH >> 6; Height = BUFFER_HEIGHT >> 6; Format = RGB10A2; };
sampler sCascadeGroup6L		{ Texture = texCascadeGroup6L; };
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

float3 gaussian_remap(float3 higher, float3 lower)
{
    float3 gaussian = lower - higher;
    float3 exposure = lower + higher;

    float sigma = textureLUT(sWeightsLUT, exposure);
    float alpha = safesqrt(_Alpha) * 2.0;

    float3 x = _M ? 1.0 - exposure : dot3((alpha > 0 ? 2 : 1) * safesqrt(gaussian));
    float3 curve = rsqrt(sigma) * x + sqrt(sigma) * exp(-sigma * abs(gaussian)) * gaussian;

    return curve * alpha;
}

float3 lpass(sampler2D s, float2 uv, float mip, float resolution)
{
    // 3x3 tap gaussian aproxximation
    float2 tap = BUFFER_PIXEL_SIZE * rcp(resolution);

    float3 a = tex2Dlod(s, float4(uv.x - tap.x, uv.y + tap.y, 0, 0));
    float3 b = tex2Dlod(s, float4(uv.x,         uv.y + tap.y, 0, 0));
    float3 c = tex2Dlod(s, float4(uv.x + tap.x, uv.y + tap.y, 0, 0));

    float3 d = tex2Dlod(s, float4(uv.x - tap.x, uv.y, 0, 0));
    float3 e = tex2Dlod(s, float4(uv.x,         uv.y, 0, 0));
    float3 f = tex2Dlod(s, float4(uv.x + tap.x, uv.y, 0, 0));

    float3 g = tex2Dlod(s, float4(uv.x - tap.x, uv.y - tap.y, 0, 0));
    float3 h = tex2Dlod(s, float4(uv.x,         uv.y - tap.y, 0, 0));
    float3 i = tex2Dlod(s, float4(uv.x + tap.x, uv.y - tap.y, 0, 0));

    float3 window = e * 4.0;
    window += (b + d + f + h) * 2.0;
    window += (a + c + g + i);

    return window / 16.0;
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
void csc_g1s(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(ReShade::BackBuffer, texcoord, 0, 0.5), 1.0);
}
void csc_g1l(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeGroup1S, texcoord, 1, 0.5), 1.0);
}
void csc_g2s(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeGroup1L, texcoord, 2, 0.25), 1.0);
}
void csc_g2l(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeGroup2S, texcoord, 3, 0.25), 1.0);
}
void csc_g3s(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeGroup2L, texcoord, 4, 0.125), 1.0);
}
void csc_g3l(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeGroup3S, texcoord, 5, 0.125), 1.0);
}
void csc_g4s(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeGroup3L, texcoord, 6, 0.0625), 1.0);
}
void csc_g4l(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeGroup4S, texcoord, 7, 0.0625), 1.0);
}
void csc_g5s(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeGroup4L, texcoord, 8, 0.03125), 1.0);
}
void csc_g5l(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeGroup5S, texcoord, 9, 0.03125), 1.0);
}
void csc_g6s(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeGroup5L, texcoord, 10, 0.015625), 1.0);
}
void csc_g6l(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) {
    o = float4(lpass(sCascadeGroup6S, texcoord, 11, 0.015625), 1.0);
}

void fetchLevel(in float2 uv, out float3 L[13])
{
    L = {
        tex2D(ReShade::BackBuffer, uv).rgb,
        tex2D(sCascadeGroup1S, uv).rgb,
        tex2D(sCascadeGroup1L, uv).rgb,
        tex2D(sCascadeGroup2S, uv).rgb,
        tex2D(sCascadeGroup2L, uv).rgb,
        tex2D(sCascadeGroup3S, uv).rgb,
        tex2D(sCascadeGroup3L, uv).rgb,
        tex2D(sCascadeGroup4S, uv).rgb,
        tex2D(sCascadeGroup4L, uv).rgb,
        tex2D(sCascadeGroup5S, uv).rgb,
        tex2D(sCascadeGroup5L, uv).rgb,
        tex2D(sCascadeGroup6S, uv).rgb,
        tex2D(sCascadeGroup6L, uv).rgb
    };
}

void main(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float3 output : SV_Target)
{
    float3 L[13];
    float3 G[13];
    float3 x = 0;

    fetchLevel(texcoord, L);

    float3 center = L[0]; // original

    float s = rsqrt(exp2(_Beta));
    float2 o = float2(1.0 - s, s);

    // cascaded gaussians, low-pass upsampling
    for (int i = 1; i <= 11; i++) 
    {
        L[i+1] = L[i] * o.x + L[i+1] * o.y;
    }

    // the second derivative...
    for (int i = 1; i <= 12; i++) 
    {
        G[i] = gaussian_remap(L[i], L[i-1]);
    }
    
    // cascaded laplacians next, high-pass upsampling
    for (int i = 1; i <= 11; i++) 
    { 
        G[i+1] = o.y * G[i] + o.x * G[i+1];
    }

    for (int i = 1; i <= 12; i++) 
    {
        x += G[i];
    }

    x /= 12.0;
    
    float2 mean;
    mean.x = dot3(center);
    mean.y = dot3(x);
    
    center = to_hdr(from_gamma(center));
    center *= (mean.x + 2.0 * mean.y) / (mean.x);
    center = to_gamma(from_hdr(center));

    output = center;
}

/*=============================================================================
/   Technique Definition
/============================================================================*/
technique UNiT_Laplacians < 
ui_label = "UNiT: Laplacians";
ui_tooltip = "									UNiT: Fast-LLF \n\n" "___________________________________________________________________________________________\n\n" "Fast and visual-accurate implementation of the local laplacian filtering image enhancement\n" "that capable in real time, similar to that used in professional photo editors.\n\n" " - Developed by RG2PS - "; >
{
    // <> Group 1
    pass { VertexShader = PostProcessVS; PixelShader = csc_g1s; RenderTarget = texCascadeGroup1S; }
    pass { VertexShader = PostProcessVS; PixelShader = csc_g1l; RenderTarget = texCascadeGroup1L; }
    // <> Group 2
    pass { VertexShader = PostProcessVS; PixelShader = csc_g2s; RenderTarget = texCascadeGroup2S; }
    pass { VertexShader = PostProcessVS; PixelShader = csc_g2l; RenderTarget = texCascadeGroup2L; }
    // <> Group 3
    pass { VertexShader = PostProcessVS; PixelShader = csc_g3s; RenderTarget = texCascadeGroup3S; }
    pass { VertexShader = PostProcessVS; PixelShader = csc_g3l; RenderTarget = texCascadeGroup3L; }
    // <> Group 4
    pass { VertexShader = PostProcessVS; PixelShader = csc_g4s; RenderTarget = texCascadeGroup4S; }
    pass { VertexShader = PostProcessVS; PixelShader = csc_g4l; RenderTarget = texCascadeGroup4L; }
    // <> Group 5
    pass { VertexShader = PostProcessVS; PixelShader = csc_g5s; RenderTarget = texCascadeGroup5S; }
    pass { VertexShader = PostProcessVS; PixelShader = csc_g5l; RenderTarget = texCascadeGroup5L; }
    // <> Group 6
    pass { VertexShader = PostProcessVS; PixelShader = csc_g6s; RenderTarget = texCascadeGroup6S; }
    pass { VertexShader = PostProcessVS; PixelShader = csc_g6l; RenderTarget = texCascadeGroup6L; }
    // <>
    pass { VertexShader = PostProcessVS; PixelShader = weight_lut; RenderTarget = texWeightsLUT; }
    pass { VertexShader = PostProcessVS; PixelShader = main; }
}
