/*
   UNiT - Shader Library for ReShade.
   Analytical Bloom via Gaussian-Guided Pyramidal Convolution
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/
 
uniform float _Intensity
<
    ui_type = "drag";
    ui_label = "Bloom Intensity";
    ui_category = "Global";
    ui_min = 0.001; ui_max = 1.0;
> = 0.5;

uniform int _GamutMap
<
    ui_type = "combo";
    ui_items = "sRGB\0ACES-AP1\0";
    ui_label = "Gamut Mapping";
    ui_category = "Global";
> = 1;

uniform int _Radiance
<
    ui_label = "Bloom  Radiance";
    ui_type = "slider";
    ui_category = "Bloom PDF Map";
    ui_tooltip = "Increases the glow intensity from point light sources.";
    ui_min = 0; ui_max = 16;
> = 10;

uniform float _MaskAmount
<
    ui_type = "slider";
    ui_label = "Depth Mask Strength";
    ui_category = "Depth Mask";
    ui_min = 0.0; ui_max = 1.0;
> = 0.0;

uniform int _DepthMode
<
    ui_type = "combo";
    ui_items = "Default\0Inverse\0";
    ui_label = "Game Depth Mode";
    ui_tooltip = "Games can use alternative depth mode. Try inverse mode if depth mask don't work properly.";
    ui_category = "Depth Mask";
> = 0;

uniform bool _Debug
<
    ui_label = "Visualize Bloom";
    ui_category = "Debug Mode";
> = false;

/*=============================================================================
/   Buffer Samplers Definition
/============================================================================*/
#include "ReShade.fxh"

texture2D texBloomMap		{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = RGBA16F; };

// Basic downsampling pass
texture2D texBloomLevel0	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = RGBA16F; };
texture2D texBloomLevel1	{ Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = RGBA16F; };
texture2D texBloomLevel2	{ Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = RGBA16F; };
texture2D texBloomLevel3	{ Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = RGBA16F; };
texture2D texBloomLevel4	{ Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = RGBA16F; };
texture2D texBloomLevel5	{ Width = BUFFER_WIDTH >> 6; Height = BUFFER_HEIGHT >> 6; Format = RGBA16F; };
texture2D texBloomLevel6	{ Width = BUFFER_WIDTH >> 7; Height = BUFFER_HEIGHT >> 7; Format = RGBA16F; };

// Laplacian decomposing pass
texture2D texFreqLevel0	    { Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = RGBA16F; };
texture2D texFreqLevel1	    { Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = RGBA16F; };
texture2D texFreqLevel2	    { Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = RGBA16F; };
texture2D texFreqLevel3	    { Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = RGBA16F; };
texture2D texFreqLevel4	    { Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = RGBA16F; };
texture2D texFreqLevel5	    { Width = BUFFER_WIDTH >> 6; Height = BUFFER_HEIGHT >> 5; Format = RGBA16F; };

// Pyramid reconstruction
texture2D texSampledLevel0	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = RGBA16F; };
texture2D texSampledLevel1	{ Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = RGBA16F; };
texture2D texSampledLevel2	{ Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = RGBA16F; };
texture2D texSampledLevel3	{ Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = RGBA16F; };
texture2D texSampledLevel4	{ Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = RGBA16F; };
texture2D texSampledLevel5	{ Width = BUFFER_WIDTH >> 6; Height = BUFFER_HEIGHT >> 6; Format = RGBA16F; };
texture2D texSampledLevel6	{ Width = BUFFER_WIDTH >> 7; Height = BUFFER_HEIGHT >> 7; Format = RGBA16F; };

// Render pyramid in half res-scale, It's doesn't affect quality much
texture2D texBloomPyramid	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = RGBA16F; };

sampler sBloomMap		    { Texture = texBloomMap; };

sampler sBloomLevel0		{ Texture = texBloomLevel0; };
sampler sBloomLevel1		{ Texture = texBloomLevel1; };
sampler sBloomLevel2		{ Texture = texBloomLevel2; };
sampler sBloomLevel3		{ Texture = texBloomLevel3; };
sampler sBloomLevel4		{ Texture = texBloomLevel4; };
sampler sBloomLevel5		{ Texture = texBloomLevel5; };
sampler sBloomLevel6		{ Texture = texBloomLevel6; };

sampler sFreqLevel0	        { Texture = texFreqLevel0; };
sampler sFreqLevel1	        { Texture = texFreqLevel1; };
sampler sFreqLevel2	        { Texture = texFreqLevel2; };
sampler sFreqLevel3	        { Texture = texFreqLevel3; };
sampler sFreqLevel4	        { Texture = texFreqLevel4; };
sampler sFreqLevel5	        { Texture = texFreqLevel5; };

sampler sSampledLevel0	    { Texture = texSampledLevel0; };
sampler sSampledLevel1	    { Texture = texSampledLevel1; };
sampler sSampledLevel2	    { Texture = texSampledLevel2; };
sampler sSampledLevel3	    { Texture = texSampledLevel3; };
sampler sSampledLevel4	    { Texture = texSampledLevel4; };
sampler sSampledLevel5	    { Texture = texSampledLevel5; };
sampler sSampledLevel6	    { Texture = texSampledLevel6; };
sampler sBloomPyramid	    { Texture = texBloomPyramid; MagFilter = LINEAR; MinFilter = LINEAR; MipFilter = LINEAR;};

/*=============================================================================
/   Global Helper Functions
/============================================================================*/
float get_lin_depth(float2 uv)
{
    return ReShade::GetLinearizedDepth(uv);
}

float3 to_linear(float3 x)
{
    return lerp(x / 12.92, pow((x + 0.055)/(1.055), 2.4), step(0.04045, x));
}

float3 to_linear_fast(float3 x)
{
    return x * x;
}

float3 from_linear_fast(float3 x)
{
    return sqrt(x);
}

float3 from_hdr(float3 x) 
{
    return saturate(x * rsqrt(1 + x * x));
}

float3 to_hdr(float3 x) 
{
   return x * rsqrt(1 - x * x + 0.003921);
}

// sRGB <-> ACESAP1 w.o RRT/ODT
float3 to_ap1(float3 x)
{
    float3x3 m = float3x3
    (
        0.613189, 0.339513, 0.047376,
        0.070207, 0.916342, 0.013451, 
        0.020618, 0.109573, 0.869609
    );

    return _GamutMap ? mul(m, x) : x;
}

float3 from_ap1(float3 x)
{
    float3x3 m = float3x3
    (
        1.705079, -0.621778, -0.083256,
       -0.130257,  1.140804, -0.010548,
       -0.024003, -0.128969,  1.152972
    );

    return _GamutMap ? mul(m, x) : x;
}

float weyl(float2 p)
{
    return frac(0.5 + p.x * 0.7548776662467 + p.y * 0.569840290998);
}

/*=============================================================================
/   Workspace Helper Functions
/============================================================================*/
int get_effective_radius(float sigma)
{
   return (int)floor(3.141 * sigma - 0.318 / sigma);
}

float4 t2c(sampler2D s, float2 uv, float L) 
{   
    float sigma = max(0.707106, sqrt(L * sqrt(2.0)));

    float2 scale = BUFFER_PIXEL_SIZE * (1 << ((int)L + 1));

    int k = get_effective_radius(sigma);

    float4 sum = 0.0;
    float total = 0.0;

    for(int i = -k; i <= k; i++) 
    for(int j = -k; j <= k; j++) 
    {
	    float2 offset = float2(i, j);
	    float2 pos = uv + offset * scale;

        bool within_bounds = all(saturate(pos) == pos);

	    float weight = exp(-dot(offset, offset) / (sigma * sigma * 2.0));
        weight *= within_bounds;

	    sum += tex2Dlod(s, float4(pos, 0, 0)) * weight;
	    total += weight;
    }

    return total > 0 ? sum / total : 0;
}

float4 t2s(sampler2D s, float2 uv, float L)
{
    float2 scale = BUFFER_PIXEL_SIZE * (1 << ((int)L));
    
    float wc = 0.41242;
    float ws = 0.14684;
    
    float4 center = tex2Dlod(s, float4(uv, 0, 0));
    
    float4 sides = 
        tex2Dlod(s, float4(uv + float2(-scale.x, 0), 0, 0)) +
        tex2Dlod(s, float4(uv + float2( scale.x, 0), 0, 0)) +
        tex2Dlod(s, float4(uv + float2(0, -scale.y), 0, 0)) +
        tex2Dlod(s, float4(uv + float2(0,  scale.y), 0, 0));
    
    return center * wc + sides * ws;
}

float4 facos_noclip(float4 x) 
{
    float4 v = abs(x);
    float4 a = sqrt(1.0 - v) * (-0.16882 * v + 1.56734);
    return x > 0.0 ? a : 3.14159 - a;
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
void pdf_map(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    float4 color, color_scale, intensity;
	float3 probability;

    color = to_linear(t2s(ReShade::BackBuffer, texcoord, 0).rgb);
    color.a = dot(color, float3(0.2126729, 0.7151522, 0.072175));
	
	color_scale = -log(1.0 - color);

    float4 shoot = facos_noclip((1.0 - color) * 6.28318) * _Radiance * _Radiance; 

    // To avoid force dynamic range compression, at simple use the NaN error to reset the value.
    if (any(isnan(shoot)) || color.a >= 0.99) shoot = 0;
	
	intensity = (color_scale - color);

    // something like: sqrt(irradiance) * absorption
	probability = sqrt(shoot.rgb + intensity.rgb * intensity.a) * color.rgb;

    output = float4(clamp(to_ap1(probability), 0.0, 65535.0), get_lin_depth(texcoord));
}

void dl_0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = t2c(sBloomMap, texcoord, 0);
}

void dl_1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
   o = t2c(sBloomLevel0, texcoord, 1);
}

void dl_2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = t2c(sBloomLevel1, texcoord, 2);
}

void dl_3(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = t2c(sBloomLevel2, texcoord, 3);
}

void dl_4(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = t2c(sBloomLevel3, texcoord, 4);
}

void dl_5(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = t2c(sBloomLevel4, texcoord, 5);
}

void dl_6(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = t2c(sBloomLevel5, texcoord, 6);
}

void fl_5(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    float4 a = tex2D(sBloomLevel5, texcoord);
    float4 b = tex2D(sBloomLevel6, texcoord);

    o = max(1e-5, a - b);
}

void fl_4(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    float4 a = tex2D(sBloomLevel4, texcoord);
    float4 b = tex2D(sBloomLevel5, texcoord);

    o = max(1e-5, a - b);
}

void fl_3(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    float4 a = tex2D(sBloomLevel3, texcoord);
    float4 b = tex2D(sBloomLevel4, texcoord);

    o = max(1e-5, a - b);
}

void fl_2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    float4 a = tex2D(sBloomLevel2, texcoord);
    float4 b = tex2D(sBloomLevel3, texcoord);

    o = max(1e-5, a - b);
}

void fl_1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    float4 a = tex2D(sBloomLevel1, texcoord);
    float4 b = tex2D(sBloomLevel2, texcoord);

    o = max(1e-5, a - b);
}

void fl_0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    float4 a = tex2D(sBloomLevel0, texcoord);
    float4 b = tex2D(sBloomLevel1, texcoord);

    o = max(1e-5, a - b);
}

void ul_6(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = t2c(sBloomLevel6, texcoord, 6);
}

void ul_5(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    float4 a = t2s(sBloomLevel5, texcoord, 5);
    float4 b = tex2D(sSampledLevel6, texcoord);

    o = (a + b) / 2.0;
}

void ul_4(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    float4 a = t2s(sBloomLevel4, texcoord, 4);
    float4 b = tex2D(sSampledLevel5, texcoord);

    o = (a + b) / 2.0;
}

void ul_3(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    float4 a = t2s(sBloomLevel3, texcoord, 3);
    float4 b = tex2D(sSampledLevel4, texcoord);

    o = (a + b) / 2.0;
}

void ul_2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    float4 a = t2s(sBloomLevel2, texcoord, 2);
    float4 b = tex2D(sFreqLevel3, texcoord);

    o = a + b;
}

void ul_1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    float4 a = t2s(sBloomLevel1, texcoord, 1);
    float4 b = tex2D(sFreqLevel2, texcoord);

    o = a + b;
}

void ul_0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    float4 a = t2s(sBloomLevel0, texcoord, 0);
    float4 b = tex2D(sFreqLevel1, texcoord);

    o = a + b;
}

void reconstruct(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    // weights are optimized to ensure pyramid quasi-continuity
    output = 
         t2s(sSampledLevel0, texcoord, 0) * 0.257126 +
         t2s(sSampledLevel1, texcoord, 1) * 0.050901 +
         t2s(sSampledLevel2, texcoord, 2) * 0.075099 +
         t2s(sSampledLevel3, texcoord, 3) * 0.386931 +
         t2s(sSampledLevel4, texcoord, 4) * 0.167947 +
         t2s(sSampledLevel5, texcoord, 5) * 0.039476 +
         t2s(sSampledLevel6, texcoord, 6) * 0.022469;
}

void get_depth_mask(inout float3 bloom, float d_avg, float d_center)
{
    float range = abs(d_avg - d_center) / d_center + 0.01;
    float m = saturate(exp2(-_MaskAmount * range));
    m = lerp(m, 1.0 - m * _MaskAmount, _DepthMode);
    bloom *= lerp(m, 1, 0.15);
}

void main(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float3 output : SV_Target)
{
    float4 bloom = tex2D(sBloomPyramid, texcoord);
    float depth = get_lin_depth(texcoord);

    get_depth_mask(bloom.rgb, bloom.w, depth);

    float3 color = tex2D(ReShade::BackBuffer, texcoord).xyz;
    
    color = to_ap1(to_linear_fast(color));

    color = to_hdr(color);
    color += bloom.rgb * _Intensity;
    color = from_hdr(color);

    color = from_linear_fast(from_ap1(color));

    // 8 bit-size
    float3 mn = floor(color * 255.0) / 255.0;
    float3 mx = ceil(color * 255.0) / 255.0;
    float3 err = saturate((color - mn) / (mx - mn));
    color = lerp(mn, mx, step(weyl(vpos.xy), err));

    output = _Debug ? from_linear_fast(from_hdr(bloom * _Intensity)) : color;
}

/*=============================================================================
/   Technique Definition
/============================================================================*/
technique UNiTLumi 
<
ui_label = "UNiT: Lumi-Bloom";
ui_tooltip = "                                  UNiT: Lumi-Bloom \n\n" "__________________________________________________________________________________________\n\n" "Lumi is the physically inspired bloom shader which simulate the realistic light diffusion.\n\n" " - Developed by RG2PS - "; >
{
    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = pdf_map;
	    RenderTarget = texBloomMap;
    }

    #define PROCESS_DOWNSAMPLE(i) pass { VertexShader = PostProcessVS; PixelShader = dl_##i; RenderTarget = texBloomLevel##i; }

    PROCESS_DOWNSAMPLE(0)
    PROCESS_DOWNSAMPLE(1)
    PROCESS_DOWNSAMPLE(2)
    PROCESS_DOWNSAMPLE(3)
    PROCESS_DOWNSAMPLE(4)
    PROCESS_DOWNSAMPLE(5)
    PROCESS_DOWNSAMPLE(6)

    #define PROCESS_LAPLACIAN(i) pass { VertexShader = PostProcessVS; PixelShader = fl_##i; RenderTarget = texFreqLevel##i; }

    PROCESS_LAPLACIAN(0)
    PROCESS_LAPLACIAN(1)
    PROCESS_LAPLACIAN(2)
    PROCESS_LAPLACIAN(3)
    PROCESS_LAPLACIAN(4)
    PROCESS_LAPLACIAN(5)

    #define PROCESS_UPSAMPLE(i) pass { VertexShader = PostProcessVS; PixelShader = ul_##i; RenderTarget = texSampledLevel##i;}

    PROCESS_UPSAMPLE(0)
    PROCESS_UPSAMPLE(1)
    PROCESS_UPSAMPLE(2)
    PROCESS_UPSAMPLE(3)
    PROCESS_UPSAMPLE(4)
    PROCESS_UPSAMPLE(5)
    PROCESS_UPSAMPLE(6)

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = reconstruct;
	    RenderTarget = texBloomPyramid;
    }

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = main;
    }
}
