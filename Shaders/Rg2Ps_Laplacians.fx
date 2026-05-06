/*
   UNiT - Shader Library for ReShade.

   Basic Implementation of the Fast Local Laplacian Filtering
   More about it: "https://hal.science/hal-01063419"
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform float A
<
    ui_label = "Contrast Strength";
    ui_type = "drag";
    ui_min = -1.0; ui_max = 1.0;
> = 1.0;

uniform int _w
<
    ui_type = "combo";
    ui_items = "Root-Two\0Gaussian\0";
    ui_label = "Laplacian Weighting";
> = 0;

/*=============================================================================
/   Buffer Samplers Definition
/============================================================================*/
#include "ReShade.fxh"

// (1, 1)
// for better performace can be rendered in half-res (1/2 -> 1/64), but I do render at full-res for a mathematically pure result
texture2D texGaussianSam1LevelCurr	{ Width = BUFFER_WIDTH >> 0; Height = BUFFER_HEIGHT >> 0; Format = RGB10A2; };
sampler sGaussianSam1LevelCurr		{ Texture = texGaussianSam1LevelCurr; };
texture2D texGaussianSam1LevelNext	{ Width = BUFFER_WIDTH >> 0; Height = BUFFER_HEIGHT >> 0; Format = RGB10A2; };
sampler sGaussianSam1LevelNext		{ Texture = texGaussianSam1LevelNext; };
// (1/2, 1/2)
texture2D texGaussianSam2LevelCurr	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = RGB10A2; };
sampler sGaussianSam2LevelCurr		{ Texture = texGaussianSam2LevelCurr; };
texture2D texGaussianSam2LevelNext	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = RGB10A2; };
sampler sGaussianSam2LevelNext		{ Texture = texGaussianSam2LevelNext; };
// (1/4, 1/4)
texture2D texGaussianSam3LevelCurr	{ Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = RGB10A2; };
sampler sGaussianSam3LevelCurr		{ Texture = texGaussianSam3LevelCurr; };
texture2D texGaussianSam3LevelNext	{ Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = RGB10A2; };
sampler sGaussianSam3LevelNext		{ Texture = texGaussianSam3LevelNext; };
// (1/8, 1/8)
texture2D texGaussianSam4LevelCurr	{ Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = RGB10A2; };
sampler sGaussianSam4LevelCurr		{ Texture = texGaussianSam4LevelCurr; };
texture2D texGaussianSam4LevelNext	{ Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = RGB10A2; };
sampler sGaussianSam4LevelNext		{ Texture = texGaussianSam4LevelNext; };
// (1/16, 1/16)
texture2D texGaussianSam5LevelCurr	{ Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = RGB10A2; };
sampler sGaussianSam5LevelCurr		{ Texture = texGaussianSam5LevelCurr; };
texture2D texGaussianSam5LevelNext	{ Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = RGB10A2; };
sampler sGaussianSam5LevelNext		{ Texture = texGaussianSam5LevelNext; };
// (1/32, 1/32)
texture2D texGaussianSam6LevelCurr	{ Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = RGB10A2; };
sampler sGaussianSam6LevelCurr		{ Texture = texGaussianSam6LevelCurr; };
texture2D texGaussianSam6LevelNext	{ Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = RGB10A2; };
sampler sGaussianSam6LevelNext		{ Texture = texGaussianSam6LevelNext; };

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
    return pow(abs(x), rcp(2.2)) * sign(x);
}

float3 from_gamma(float3 x)
{
    return pow(abs(x), 2.2) * sign(x);
}

float3 to_hdr(float3 x)
{
    return x * rsqrt(1.0 - x * x + rcp(255.0));
}

float3 from_hdr(float3 x)
{
    return x * rsqrt(1.0 + x * x);
}

float clampn1(float x)
{
    return clamp(x, -1.0, 1.0);
}

/*=============================================================================
/   Workspace Helper Functions
/============================================================================*/
#define SIGMA_WEIGHT 20.0

/*
// f(g) = (1/o) * g + exp(-o * |g|) * g * alpha
float3 do_remap(float3 x, float3 center)
{
    const float o = SIGMA_WEIGHT;

    float3 gaussian = center - x;
    
    return rcp(o) * gaussian + exp(-o * abs(gaussian)) * gaussian * clampn1(A);
}
*/

/*
// f(g) = (1/o * (1/|alpha|)) * g + exp(-(o * (1/|alpha|)) * |g|) * g * sgn(alpha)
float3 do_remap(float3 x, float3 center)
{
    float o = SIGMA_WEIGHT / (A == 0 ? 1.0 : abs(A));

    float3 gaussian = center - x;

    return rcp(o) * gaussian + exp(-o * abs(gaussian)) * gaussian * (A > 0.0 ? sign(A) : A);
}
*/

float3 do_remap(float3 x, float3 center)
{
    float o = SIGMA_WEIGHT;
    
    float3 gaussian = center - x;

    return rcp(o) * gaussian + exp(-o * abs(gaussian)) * gaussian * clampn1(A);
}

float3 downsample(sampler2D s, float2 uv, float resolution)
{
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

    return (e*4.0 + (a+b+c+d)*2.0 + (f+g+h+i)) * 0.0625;
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
void t1_curr(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(downsample(ReShade::BackBuffer, texcoord, 1.0), 1.0);
}

void t1_temp(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(downsample(sGaussianSam1LevelCurr, texcoord, 1.0), 1.0);
}

void t2_curr(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(downsample(sGaussianSam1LevelNext, texcoord, 0.5), 1.0);
}

void t2_temp(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(downsample(sGaussianSam2LevelCurr, texcoord, 0.5), 1.0);
}

void t3_curr(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(downsample(sGaussianSam2LevelNext, texcoord, 0.25), 1.0);
}

void t3_temp(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(downsample(sGaussianSam3LevelCurr, texcoord, 0.25), 1.0);
}

void t4_curr(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(downsample(sGaussianSam3LevelNext, texcoord, 0.125), 1.0);
}

void t4_temp(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(downsample(sGaussianSam4LevelCurr, texcoord, 0.125), 1.0);
}

void t5_curr(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(downsample(sGaussianSam4LevelNext, texcoord, 0.0625), 1.0);
}

void t5_temp(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(downsample(sGaussianSam5LevelCurr, texcoord, 0.0625), 1.0);
}

void t6_curr(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(downsample(sGaussianSam5LevelNext, texcoord, 0.03125), 1.0);
}

void t6_temp(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(downsample(sGaussianSam6LevelCurr, texcoord, 0.03125), 1.0);
}

void gaussianfetch(in float2 uv, out float3 _sample[13])
{
    _sample = 
    {
        tex2D(ReShade::BackBuffer,    uv).rgb,
        tex2D(sGaussianSam1LevelCurr, uv).rgb,
        tex2D(sGaussianSam1LevelNext, uv).rgb,
        tex2D(sGaussianSam2LevelCurr, uv).rgb,
        tex2D(sGaussianSam2LevelNext, uv).rgb,
        tex2D(sGaussianSam3LevelCurr, uv).rgb,
        tex2D(sGaussianSam3LevelNext, uv).rgb,
        tex2D(sGaussianSam4LevelCurr, uv).rgb,
        tex2D(sGaussianSam4LevelNext, uv).rgb,
        tex2D(sGaussianSam5LevelCurr, uv).rgb,
        tex2D(sGaussianSam5LevelNext, uv).rgb,
        tex2D(sGaussianSam6LevelCurr, uv).rgb,
        tex2D(sGaussianSam6LevelNext, uv).rgb
    };
}

void main(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float3 output : SV_Target)
{
    float3 p[13];
    float3 v[13];
    float3 x = 0;

    gaussianfetch(texcoord, p);

    float2 weight = _w ? float2(0.377541, 0.622459) : float2(0.292893, 0.707106);

    // low-pass upsampling -> (forward)
    for (int i = 1; i <= 11; i++) 
    {
        p[i+1] = p[i] * weight.x + p[i+1] * weight.y;
    }

    // transform
    for (int i = 1; i <= 12; i++) 
    {
        v[i] = do_remap(p[i], p[i-1]);
    }
    
    // high-pass upsampling <- (backward)
    for (int i = 1; i <= 11; i++) 
    { 
        v[i+1] = weight.y * v[i] + weight.x * v[i+1];
    }

    for (int i = 1; i <= 11; i++) 
    {
        x += v[i];
    }
    
    float3 color = p[0];
    float luma = dot3(x);

    float center_luma = dot3(color);
    float linear_luma = luma * 2.0;
    
    color = to_hdr(from_gamma(color));
    color *= (center_luma + linear_luma) / (center_luma + 1e-6);
    color = to_gamma(from_hdr(color));

    output = color;
}

/*=============================================================================
/   Technique Definition
/============================================================================*/
technique UNiT_Laplacians < 
ui_label = "UNiT: Laplacians";
ui_tooltip = "									UNiT: Fast-LLF \n\n" "___________________________________________________________________________________________\n\n" "Fast and visual-accurate implementation of the local laplacian filtering image enhancement\n" "that capable in real time, similar to that used in professional photo editors.\n\n" " - Developed by RG2PS - "; >
{
    // <> Group 1
    pass { VertexShader = PostProcessVS; PixelShader = t1_curr; RenderTarget = texGaussianSam1LevelCurr; }
    pass { VertexShader = PostProcessVS; PixelShader = t1_temp; RenderTarget = texGaussianSam1LevelNext; }
    // <> Group 2
    pass { VertexShader = PostProcessVS; PixelShader = t2_curr; RenderTarget = texGaussianSam2LevelCurr; }
    pass { VertexShader = PostProcessVS; PixelShader = t2_temp; RenderTarget = texGaussianSam2LevelNext; }
    // <> Group 3
    pass { VertexShader = PostProcessVS; PixelShader = t3_curr; RenderTarget = texGaussianSam3LevelCurr; }
    pass { VertexShader = PostProcessVS; PixelShader = t3_temp; RenderTarget = texGaussianSam3LevelNext; }
    // <> Group 4
    pass { VertexShader = PostProcessVS; PixelShader = t4_curr; RenderTarget = texGaussianSam4LevelCurr; }
    pass { VertexShader = PostProcessVS; PixelShader = t4_temp; RenderTarget = texGaussianSam4LevelNext; }
    // <> Group 5
    pass { VertexShader = PostProcessVS; PixelShader = t5_curr; RenderTarget = texGaussianSam5LevelCurr; }
    pass { VertexShader = PostProcessVS; PixelShader = t5_temp; RenderTarget = texGaussianSam5LevelNext; }
    // <> Group 6
    pass { VertexShader = PostProcessVS; PixelShader = t6_curr; RenderTarget = texGaussianSam6LevelCurr; }
    pass { VertexShader = PostProcessVS; PixelShader = t6_temp; RenderTarget = texGaussianSam6LevelNext; }
    // <>
    pass { VertexShader = PostProcessVS; PixelShader = main; }
}