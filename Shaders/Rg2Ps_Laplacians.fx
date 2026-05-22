/*
   UNiT - Shader Library for ReShade.

   Implementation of the Fast Local Laplacian Filtering
   More about it: "https://hal.science/hal-01063419"
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform float ALPHA
<
    ui_label = "Contrast Strength";
    ui_type = "drag";
    ui_min = -1.0; ui_max = 1.0;
> = 0.75;

/*=============================================================================
/   Buffer Samplers Definition
/============================================================================*/
#include "ReShade.fxh"

// (1, 1)
// for better performace can be rendered in half-res (1/2 -> 1/64), but I do render at full-res for a mathematically pure result
texture2D texGaussianSam1Level	{ Width = BUFFER_WIDTH >> 0; Height = BUFFER_HEIGHT >> 0; Format = RGB10A2; };
sampler sGaussianSam1Level		{ Texture = texGaussianSam1Level; };
// (1/2, 1/2)
texture2D texGaussianSam2Level	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = RGB10A2; };
sampler sGaussianSam2Level		{ Texture = texGaussianSam2Level; };
// (1/4, 1/4)
texture2D texGaussianSam3Level	{ Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = RGB10A2; };
sampler sGaussianSam3Level		{ Texture = texGaussianSam3Level; };
// (1/8, 1/8)
texture2D texGaussianSam4Level	{ Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = RGB10A2; };
sampler sGaussianSam4Level		{ Texture = texGaussianSam4Level; };
// (1/16, 1/16)
texture2D texGaussianSam5Level	{ Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = RGB10A2; };
sampler sGaussianSam5Level		{ Texture = texGaussianSam5Level; };
// (1/32, 1/32)
texture2D texGaussianSam6Level	{ Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = RGB10A2; };
sampler sGaussianSam6Level		{ Texture = texGaussianSam6Level; };

/*=============================================================================
/   Global Helper Functions
/============================================================================*/
float luminance(float3 x)
{
    return dot(x, float3(0.299, 0.587, 0.114));
}

float3 to_hdr(float3 x)
{
    return -log2(1.0 - min(0.9999999, x / 1.1));
}

float3 from_hdr(float3 x)
{
    return clamp((1.0 - exp2(-x)) * 1.1, -1.0, 1.0);
}

/*=============================================================================
/   Workspace Helper Functions
/============================================================================*/
float3 fx(float3 x)
{
    return (sin(x) * sin(x)) / abs(x + 1e-9);
}

float3 do_remap(in float3 x, float sigma)
{
	const float HALF_PI = 1.57079;

    float alpha = ALPHA < 0.0 ? ALPHA / 2.0 : ALPHA;
	
	float s = sigma * 2.8284;
	float s2 = s * s;
	
	float3 ax = min(abs(x) * s2, HALF_PI);

	float3 y = fx(ax);
	
	return sign(x) * y / s2 * sqrt(s) * alpha;
}

float3 sp(sampler2D s, float2 uv, float N)
{
    float2 offset = BUFFER_PIXEL_SIZE / N;
    float x = offset.x;
    float y = offset.y;

    float3 a = tex2Dlod(s, float4(uv.x - 2*x, uv.y + 2*y, 0, 0)).rgb;
    float3 b = tex2Dlod(s, float4(uv.x,       uv.y + 2*y, 0, 0)).rgb;
    float3 c = tex2Dlod(s, float4(uv.x + 2*x, uv.y + 2*y, 0, 0)).rgb;
    float3 d = tex2Dlod(s, float4(uv.x - 2*x, uv.y, 0, 0)).rgb;
    float3 e = tex2Dlod(s, float4(uv.x,       uv.y, 0, 0)).rgb;
    float3 f = tex2Dlod(s, float4(uv.x + 2*x, uv.y, 0, 0)).rgb;
    float3 g = tex2Dlod(s, float4(uv.x - 2*x, uv.y - 2*y, 0, 0)).rgb;
    float3 h = tex2Dlod(s, float4(uv.x,       uv.y - 2*y, 0, 0)).rgb;
    float3 i = tex2Dlod(s, float4(uv.x + 2*x, uv.y - 2*y, 0, 0)).rgb;
    float3 j = tex2Dlod(s, float4(uv.x - x, uv.y + y, 0, 0)).rgb;
    float3 k = tex2Dlod(s, float4(uv.x + x, uv.y + y, 0, 0)).rgb;
    float3 l = tex2Dlod(s, float4(uv.x - x, uv.y - y, 0, 0)).rgb;
    float3 m = tex2Dlod(s, float4(uv.x + x, uv.y - y, 0, 0)).rgb;
    
    // todo: there is probably no need for 13 points to cover the first level, so this can be replaced with a cross or box approximation
    return e*0.125 + (a+c+g+i)*0.03125 + (b+d+f+h)*0.0625 + (j+k+l+m)*0.125;
}

float4 cubic(float v) 
{
    float4 s, w;
    
    s.x = (1.0 - v) * (1.0 - v) * (1.0 - v);
    s.y = (2.0 - v) * (2.0 - v) * (2.0 - v);
    s.z = (3.0 - v) * (3.0 - v) * (3.0 - v);
    s.w = (4.0 - v) * (4.0 - v) * (4.0 - v);
    
    w.x = s.x;
    w.y = s.y - 4.0 * s.x;
    w.z = s.z - 4.0 * s.y + 6.0 * s.x;
    w.w = 6.0 - w.x - w.y - w.z;

    return w / 6.0;
}

float3 ssp(sampler2D s, float2 coord, float N)
{
    float2 size = BUFFER_SCREEN_SIZE / (N * 1.4142);
 
    float2 uv = coord * size - 0.5;
    float2 fxy = frac(uv);
    uv -= fxy;
 
    float4 xcubic = cubic(fxy.x);
    float4 ycubic = cubic(fxy.y);
 
    float4 c = uv.xxyy + float2(-0.5, 1.5).xyxy;
    float4 sw = float4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    float4 offset = c + float4(xcubic.yw, ycubic.yw) / sw;
 
    offset /= size.xxyy;
 
    float3 s0 = tex2Dlod(s, float4(offset.xz, 0, 0));
    float3 s1 = tex2Dlod(s, float4(offset.yz, 0, 0));
    float3 s2 = tex2Dlod(s, float4(offset.xw, 0, 0));
    float3 s3 = tex2Dlod(s, float4(offset.yw, 0, 0));
 
    float sx = sw.x / (sw.x + sw.y);
    float sy = sw.z / (sw.z + sw.w);

    return lerp(lerp(s3, s2, sx), lerp(s1, s0, sx), sy);
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
void t1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(sp(ReShade::BackBuffer, texcoord, 1.0), 1.0);
}

void t2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(sp(sGaussianSam1Level, texcoord, 0.5), 1.0);
}

void t3(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(sp(sGaussianSam2Level, texcoord, 0.25), 1.0);
}

void t4(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(sp(sGaussianSam3Level, texcoord, 0.125), 1.0);
}

void t5(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(sp(sGaussianSam4Level, texcoord, 0.0625), 1.0);
}
void t6(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target) 
{
    o = float4(sp(sGaussianSam5Level, texcoord, 0.03125), 1.0);
}

void main(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float3 output : SV_Target)
{
    float3 p[7];

    float3 center = tex2Dfetch(ReShade::BackBuffer, vpos.xy, 0).rgb;

    p[0] = center, 
    p[1] = ssp(sGaussianSam1Level, uv, 1).rgb,  // 1
    p[2] = ssp(sGaussianSam2Level, uv, 2).rgb,  // 2
    p[3] = ssp(sGaussianSam3Level, uv, 3).rgb,  // 3
    p[4] = ssp(sGaussianSam4Level, uv, 4).rgb,  // 4
    p[5] = ssp(sGaussianSam5Level, uv, 5).rgb,  // 5
    p[6] = ssp(sGaussianSam6Level, uv, 6).rgb;  // 6

    for (int i = 0; i <= 6; i++)
    {
        p[i] = to_hdr(p[i]);
    }

    float3 guide = 0;
    float ws = 0;

    for (int i = 0; i < 6; i++) 
    {
        float w = exp2(-float(i) / 0.5);
        guide += p[i] * w;
        ws += w;
    }

    guide /= ws;

    float3 signal = guide;

    for (int i = 5; i >= 0; i--)
    {
        float w = exp(float(i) * 0.5);
        
        float3 reference = do_remap(p[i] - p[i+1], w);
        
        float L0 = luminance(signal);
        float L1 = luminance(reference);
        
        signal *= (L0 + L1) / L0;
    }

    output = from_hdr(signal);
}

/*=============================================================================
/   Technique Definition
/============================================================================*/
technique UNiT_Laplacians < 
ui_label = "UNiT: Laplacians";
ui_tooltip = "						  UNiT: Fast Local Laplacians \n\n" "________________________________________________________________________________\n\n" "Implementation of the fast local laplacian filtering that capable in real time,\n" "similar to that used in professional photo editors for image processing.\n\n" " - Developed by RG2PS - "; >
{
    pass { VertexShader = PostProcessVS; PixelShader = t1; RenderTarget = texGaussianSam1Level; }
    pass { VertexShader = PostProcessVS; PixelShader = t2; RenderTarget = texGaussianSam2Level; }
    pass { VertexShader = PostProcessVS; PixelShader = t3; RenderTarget = texGaussianSam3Level; }
    pass { VertexShader = PostProcessVS; PixelShader = t4; RenderTarget = texGaussianSam4Level; }
    pass { VertexShader = PostProcessVS; PixelShader = t5; RenderTarget = texGaussianSam5Level; }
    pass { VertexShader = PostProcessVS; PixelShader = t6; RenderTarget = texGaussianSam6Level; }
    pass { VertexShader = PostProcessVS; PixelShader = main; }
}