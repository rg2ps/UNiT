/*
   UNiT - Shader Library for ReShade.
   Analytical Bloom via Gaussian Pyramid Composition
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/
 
uniform float _Intensity
<
    ui_type = "drag";
    ui_label = "Bloom Intensity";
    ui_category = "Globals";
    ui_min = 0.001; ui_max = 1.0;
> = 0.3;

uniform float _UIMask
<
    ui_type = "slider";
    ui_label = "Strength";
    ui_category = "UI Mask";
    ui_min = 0.0; ui_max = 1.0;
> = 0.0;

uniform int _ZMode
<
    ui_type = "combo";
    ui_items = "Forward\0Inverse\0";
    ui_label = "UI Depth Mode";
    ui_category = "UI Mask";
> = 0;

uniform bool _Debug
<
    ui_label = "Visualize Irradiance Map";
    ui_category = "Debug Mode";
> = false;

/*=============================================================================
/   Buffer Samplers Definition
/============================================================================*/
#include "ReShade.fxh"

#ifndef DITHER_BIT_DEPTH
    #define DITHER_BIT_DEPTH 8
#endif

#define DO_TEXTURE(N) texture2D texLowPassLevel##N { Width = BUFFER_WIDTH >> N + 1; Height = BUFFER_HEIGHT >> N + 1; Format = RGBA16F;  }; sampler sLowPassLevel##N { Texture = texLowPassLevel##N;  };  

DO_TEXTURE(0)
DO_TEXTURE(1)
DO_TEXTURE(2)
DO_TEXTURE(3)
DO_TEXTURE(4)
DO_TEXTURE(5)
DO_TEXTURE(6)

texture2D texBloomMap		
{ 
    Width = BUFFER_WIDTH >> 1; 
    Height = BUFFER_HEIGHT >> 1; 
    Format = RGBA16F; 
};

sampler sBloomMap 
{ 
    Texture = texBloomMap; 
};

texture2D texBloomPyramid	
{ 
    Width = BUFFER_WIDTH; 
    Height = BUFFER_HEIGHT; 
    Format = RGBA16F; 
};

sampler sBloomPyramid	    
{ 
    Texture = texBloomPyramid; 
};

/*=============================================================================
/   Global Helper Functions
/============================================================================*/
float get_lin_depth(float2 uv)
{
    return ReShade::GetLinearizedDepth(uv);
}

float grayscale(float3 x)
{
    return dot(x, float3(0.2126729, 0.7151522, 0.072175));
}

float3 from_hdr(float3 x) 
{
    return x * rsqrt(1.0 + x * x);
}

float3 to_hdr(float3 x) 
{
   return x * rsqrt(1.0 - x * x + rcp(255.0));
}

float rnd(float2 p)
{
    return frac(0.5 + p.x * 0.7548776662467 + p.y * 0.569840290998);
}

/*=============================================================================
/   Workspace Helper Functions
/============================================================================*/
float4 tex2Dscale(sampler2D s, float2 uv, float mip) 
{   
    float4 sum = 0.0;
    float total = 0.0;

    float sigma = exp2(0.5 * mip + 1) * rsqrt(2.0);
    float2 mip_texel = BUFFER_PIXEL_SIZE * exp2(mip + 1);

    for(int x = -ceil(sigma); x <= ceil(sigma); x++) 
    for(int y = -ceil(sigma); y <= ceil(sigma); y++) 
    {
	    float2 offset = float2(x, y);
	    float2 tap_uv = uv + offset * mip_texel;
        
        float weight = exp(-dot(offset, offset) / sigma);

	    sum += tex2Dlod(s, float4(tap_uv, 0, 0)) * weight;
	    total += weight;
    }

    return total > 0 ? sum / total : 0;
}

float4 tex2Dsample(sampler2D s, float2 uv, float mip)
{
    float2 tap = BUFFER_PIXEL_SIZE * exp2(mip + 1);
    
    float4 center = tex2Dlod(s, float4(uv, 0, 0));
    
    float4 cross = 
        tex2Dlod(s, float4(uv + float2(-tap.x, 0), 0, 0)) +
        tex2Dlod(s, float4(uv + float2( tap.x, 0), 0, 0)) +
        tex2Dlod(s, float4(uv + float2(0, -tap.y), 0, 0)) +
        tex2Dlod(s, float4(uv + float2(0,  tap.y), 0, 0));
    
    return center * 0.41242 + cross * 0.14684;
}

void depthmask(inout float3 x, float z0, float z1, float a)
{
    float v = saturate(exp2(-a * abs(z0 - z1) / z1 + 0.01));
    x *= lerp(lerp(v, 1.0 - v * a, _ZMode), 1.0, 0.15);
}

void pixeldither(in float seed, inout float3 x)
{
    float bit_depth = exp2(DITHER_BIT_DEPTH) - 1;

    float3 qu_min = floor(x * bit_depth) / bit_depth;
    float3 qu_max = ceil(x * bit_depth) / bit_depth;

    float3 error = saturate((x - qu_min) / (qu_max - qu_min));

    x = lerp(qu_min, qu_max, step(seed, error));
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
float3 chromascale(float3 x)
{
    return max(0.0, (x * x) / grayscale(x));
}

/*
    Writes irradiance map for bloom propagation
*/
void pdf_map(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target0)
{
    float3 sdr = tex2Dsample(ReShade::BackBuffer, texcoord, -2).rgb;
    float3 hdr = pow(sdr, 2.2);

    const float gamma = 0.05;

    // https://www.ipol.im/pub/art/2020/300/article.pdf
    float3 v = saturate(grayscale(sdr)); 
    float3 k = -(0.63662 - 0.63662*pow((1.0 - v) / 0.5, gamma)); 
    float3 m = 255.0f / log10(abs(k) * 255.0 + 1.0);
    float3 x = max(0.0, (255.0f - m) * (abs(k) * hdr));
    
    output = float4(chromascale(x), get_lin_depth(texcoord));
}

/*
    Start the low-pass composing: L_k = (S_k ∘ S_[k-1] ∘ ... ∘ S₀)(P)
*/
void dl_0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = tex2Dscale(sBloomMap, texcoord, 0);
}

void dl_1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
   o = tex2Dscale(sLowPassLevel0, texcoord, 1);
}

void dl_2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = tex2Dscale(sLowPassLevel1, texcoord, 2);
}

void dl_3(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = tex2Dscale(sLowPassLevel2, texcoord, 3);
}

void dl_4(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = tex2Dscale(sLowPassLevel3, texcoord, 4);
}

void dl_5(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = tex2Dscale(sLowPassLevel4, texcoord, 5);
}

void dl_6(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = tex2Dscale(sLowPassLevel5, texcoord, 6);
}

/*
    End of composing, start the upsampling: { L_k + R_[k] .. G_k + H_[k] }
*/
void ul_5(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = tex2Dsample(sLowPassLevel6, texcoord, 6);
}

void ul_4(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = tex2Dsample(sLowPassLevel5, texcoord, 5);
}

void ul_3(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = tex2Dsample(sLowPassLevel4, texcoord, 4);
}

void ul_2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = tex2Dsample(sLowPassLevel3, texcoord, 3);
}

void ul_1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = tex2Dsample(sLowPassLevel2, texcoord, 2);
}

void ul_0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = tex2Dsample(sLowPassLevel1, texcoord, 1);
}

void reconstruct(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    output = 
        tex2D(sLowPassLevel0, texcoord) * 0.285714 +
        tex2D(sLowPassLevel1, texcoord) * 0.238095 +
        tex2D(sLowPassLevel2, texcoord) * 0.190476 +
        tex2D(sLowPassLevel3, texcoord) * 0.142857 +
        tex2D(sLowPassLevel4, texcoord) * 0.095238 +
        tex2D(sLowPassLevel5, texcoord) * 0.047619;
}

void main(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float3 output : SV_Target)
{
    float3 color = tex2D(ReShade::BackBuffer, texcoord).xyz;
    float3 oc = color;
    float4 bloom = tex2D(sBloomPyramid, texcoord);

    depthmask(bloom.rgb, bloom.w, get_lin_depth(texcoord), _UIMask);
    
    color = pow(color, 2.2);

    color = to_hdr(color);
    color += bloom.rgb * _Intensity;
    color = from_hdr(color);

    color = pow(color, rcp(2.2));

    pixeldither(rnd(vpos.xy), color);

    output = _Debug ? sqrt(tex2D(sBloomMap, texcoord).rgb) : color;
}

/*=============================================================================
/   Technique Definition
/============================================================================*/
technique UNiTLumi 
<
ui_label = "UNiT: Lumi-Bloom";
ui_tooltip = "                                  UNiT: Lumi-Bloom \n\n" "__________________________________________________________________________________________\n\n" "Lumi is the physically inspired bloom shader which simulate the realistic light diffusion.\n\n" " - Developed by RG2PS - "; >
{
    // <>
    pass { VertexShader = PostProcessVS; PixelShader = pdf_map; RenderTarget = texBloomMap; }
    // <>
    pass { VertexShader = PostProcessVS; PixelShader = dl_0; RenderTarget = texLowPassLevel0; }
    pass { VertexShader = PostProcessVS; PixelShader = dl_1; RenderTarget = texLowPassLevel1; }
    pass { VertexShader = PostProcessVS; PixelShader = dl_2; RenderTarget = texLowPassLevel2; }
    pass { VertexShader = PostProcessVS; PixelShader = dl_3; RenderTarget = texLowPassLevel3; }
    pass { VertexShader = PostProcessVS; PixelShader = dl_4; RenderTarget = texLowPassLevel4; }
    pass { VertexShader = PostProcessVS; PixelShader = dl_5; RenderTarget = texLowPassLevel5; }
    pass { VertexShader = PostProcessVS; PixelShader = dl_6; RenderTarget = texLowPassLevel6; }
    // <>
    pass { VertexShader = PostProcessVS; PixelShader = ul_5; RenderTarget = texLowPassLevel5; BlendEnable = true; BlendOp = ADD; SrcBlend = ONE; DestBlend = ONE; }
    pass { VertexShader = PostProcessVS; PixelShader = ul_4; RenderTarget = texLowPassLevel4; BlendEnable = true; BlendOp = ADD; SrcBlend = ONE; DestBlend = ONE; }
    pass { VertexShader = PostProcessVS; PixelShader = ul_3; RenderTarget = texLowPassLevel3; BlendEnable = true; BlendOp = ADD; SrcBlend = ONE; DestBlend = ONE; }
    pass { VertexShader = PostProcessVS; PixelShader = ul_2; RenderTarget = texLowPassLevel2; BlendEnable = true; BlendOp = ADD; SrcBlend = ONE; DestBlend = ONE; }
    pass { VertexShader = PostProcessVS; PixelShader = ul_1; RenderTarget = texLowPassLevel1; BlendEnable = true; BlendOp = ADD; SrcBlend = ONE; DestBlend = ONE; }
    pass { VertexShader = PostProcessVS; PixelShader = ul_0; RenderTarget = texLowPassLevel0; BlendEnable = true; BlendOp = ADD; SrcBlend = ONE; DestBlend = ONE; }
    // <>
    pass { VertexShader = PostProcessVS; PixelShader = reconstruct; RenderTarget = texBloomPyramid; }
    // <>
    pass { VertexShader = PostProcessVS; PixelShader = main; }
}
