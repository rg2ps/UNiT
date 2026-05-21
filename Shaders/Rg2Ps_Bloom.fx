/*
   UNiT - Shader Library for ReShade.
   Analytical Bloom via Gaussian Pyramid Composition
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/
 
uniform float INTENSITY
<
    ui_type = "drag";
    ui_label = "Intensity";
    ui_category = "Globals";
    ui_min = 0.001; ui_max = 1.0;
> = 0.5;

uniform float SENSITIVITY
<
    ui_type = "drag";
    ui_label = "Sensitivity";
    ui_category = "Globals";
    ui_min = 0.001; ui_max = 1.0;
> = 0.75;

uniform float UIMASK
<
    ui_type = "slider";
    ui_label = "Strength";
    ui_category = "UI Mask";
    ui_min = 0.0; ui_max = 1.0;
> = 0.0;

uniform int DEPTH_MODE
<
    ui_type = "combo";
    ui_items = "Forward\0Inverse\0";
    ui_label = "UI Depth Mode";
    ui_category = "UI Mask";
> = 0;

/*=============================================================================
/   Buffer Samplers Definition
/============================================================================*/
#include "ReShade.fxh"

#ifndef DITHER_BIT_DEPTH
    #define DITHER_BIT_DEPTH 8
#endif

#define WRITE_BUFFER(N) texture2D texBloomCascade##N { Width = BUFFER_WIDTH >> N + 1; Height = BUFFER_HEIGHT >> N + 1; Format = RGBA16F;  }; sampler sBloomCascade##N { Texture = texBloomCascade##N;  };  

WRITE_BUFFER(0)
WRITE_BUFFER(1)
WRITE_BUFFER(2)
WRITE_BUFFER(3)
WRITE_BUFFER(4)
WRITE_BUFFER(5)
WRITE_BUFFER(6)

texture2D texBloomGenerate	
{ 
    Width = BUFFER_WIDTH >> 1; 
    Height = BUFFER_HEIGHT >> 1; 
    Format = RGBA16F; 
};

sampler sBloomGenerate
{ 
    Texture = texBloomGenerate;
};

texture2D texBloomFetch	
{ 
    Width = BUFFER_WIDTH; 
    Height = BUFFER_HEIGHT; 
    Format = RGBA16F; 
};

sampler sBloomFetch	    
{ 
    Texture = texBloomFetch; 
};

/*=============================================================================
/   Global Helper Functions
/============================================================================*/
float get_depth(float2 uv)
{
    return (!DEPTH_MODE ? ReShade::GetLinearizedDepth(uv) : 1.0 / ReShade::GetLinearizedDepth(uv)) + 1e-6;
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

float3 normsq(float3 x)
{
    return x * x * rsqrt(dot(x, x) / 3.0 + 1e-10);
}

/*=============================================================================
/   Workspace Helper Functions
/============================================================================*/
float4 d0(sampler2D s, float2 uv, float mip) 
{   
    float4 sum = 0.0;
    float wsum = 0.0;

    float sigma = exp2(mip / 2.0);
    float2 mip_texel = BUFFER_PIXEL_SIZE * exp2(mip);

    int r = ceil(sigma);

    for(int x = -r; x <= r; x++) 
    for(int y = -r; y <= r; y++) 
    {
	    float2 offset = float2(x, y);
	    float2 scoord = uv + offset * mip_texel;
	    
        float weight = exp(-dot(offset, offset) / sigma);

	    sum += tex2Dlod(s, float4(scoord, 0, 0)) * weight;
	    wsum += weight;
    }

    return wsum > 0 ? sum / wsum : 0;
}

float4 d1(sampler2D s, float2 uv, float mip)
{
    float2 offset = 1.5 * BUFFER_PIXEL_SIZE * exp2(mip);

    float4 a = tex2Dlod(s, float4(uv + float2(-offset.x, 0), 0, 0));
    float4 b = tex2Dlod(s, float4(uv + float2( offset.x, 0), 0, 0));
    float4 c = tex2Dlod(s, float4(uv + float2(0, -offset.y), 0, 0));
    float4 d = tex2Dlod(s, float4(uv + float2(0,  offset.y), 0, 0));
    float4 e = tex2Dlod(s, float4(uv, 0, 0));
    
    return e * 0.41242 + (a + b + c + d) * 0.14684;
}

void applyUI(inout float3 x, float center, float average, float alpha)
{
    x *= saturate(exp2(-alpha * (abs(center - average) / average) / 1000.0));
}

void do_dither(in float v, inout float3 x)
{
    float bit = exp2(DITHER_BIT_DEPTH) - 1;
    
    float3 qumin = floor(x * bit) / bit;
    float3 qumax =  ceil(x * bit) / bit;

    float3 threshold = saturate((x - qumin) / (qumax - qumin));

    x = lerp(qumin, qumax, step(v, threshold));
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
/*
    Writes irradiance map for bloom propagation
*/
void psfmap(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target0)
{
	// subpixel (atomsized) offset to prevent degenerate (noisy) pixels from being included in the calculation
	// this reduces the accumulation error at each light propagation level.
    float3 sdr = d1(ReShade::BackBuffer, texcoord, -32.0).rgb;
    float3 hdr = pow(sdr, 2.2);

    // https://www.ipol.im/pub/art/2020/300/article.pdf
    const float gamma = 0.05 * SENSITIVITY;

    float3 v = saturate(grayscale(sdr)); 
    float3 k = -(0.5 - 0.5*pow((1.0 - v) / 0.5, gamma)); 
    float3 m = 255.0f / log10(abs(k) * 255.0 + 1.0);
    float3 P = max(0.0, (sdr - abs(k) / gamma) * gamma);
    float3 x = max(0.0, (255.0f - m) * (abs(k) * hdr));
    
    output = float4(P + normsq(x), get_depth(texcoord));
}

/*
    Start the low-pass composing: L_k = (S_k ∘ S_[k-1] ∘ ... ∘ S₀)(P)
*/
void decomp0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = d0(sBloomGenerate, texcoord, 1);
}

void decomp1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
   o = d0(sBloomCascade0, texcoord, 2);
}

void decomp2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = d0(sBloomCascade1, texcoord, 3);
}

void decomp3(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = d0(sBloomCascade2, texcoord, 4);
}

void decomp4(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = d0(sBloomCascade3, texcoord, 5);
}

void decomp5(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = d0(sBloomCascade4, texcoord, 6);
}

void decomp6(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = d0(sBloomCascade5, texcoord, 7);
}

/*
    End of composing, start the upsampling: { L_k + R_[k] .. G_k + H_[k] }
*/
void recomp5(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = d1(sBloomCascade6, texcoord, 7);
}

void recomp4(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = d1(sBloomCascade5, texcoord, 6);
}

void recomp3(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = d1(sBloomCascade4, texcoord, 5);
}

void recomp2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = d1(sBloomCascade3, texcoord, 4);
}

void recomp1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = d1(sBloomCascade2, texcoord, 3);
}

void recomp0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 o : SV_Target)
{
    o = d1(sBloomCascade1, texcoord, 2);
}

void bandfetch(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    output = 
        tex2D(sBloomCascade0, texcoord) * 0.285714 +
        tex2D(sBloomCascade1, texcoord) * 0.238095 +
        tex2D(sBloomCascade2, texcoord) * 0.190476 +
        tex2D(sBloomCascade3, texcoord) * 0.142857 +
        tex2D(sBloomCascade4, texcoord) * 0.095238 +
        tex2D(sBloomCascade5, texcoord) * 0.047619;
}

void main(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float3 output : SV_Target)
{
    float3 color = tex2D(ReShade::BackBuffer, texcoord).xyz;
    float4 bloom = tex2D(sBloomFetch, texcoord);

    applyUI(bloom.rgb, bloom.w, get_depth(texcoord), UIMASK);
    
    color = pow(color, 2.2);

    color = to_hdr(color);
    color += bloom.rgb * INTENSITY;
    color = from_hdr(color);

    color = pow(color, rcp(2.2));

    do_dither(rnd(vpos.xy), color);

    output = color;
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
    pass { VertexShader = PostProcessVS; PixelShader = psfmap; RenderTarget = texBloomGenerate; }
    // <>
    pass { VertexShader = PostProcessVS; PixelShader = decomp0; RenderTarget = texBloomCascade0; }
    pass { VertexShader = PostProcessVS; PixelShader = decomp1; RenderTarget = texBloomCascade1; }
    pass { VertexShader = PostProcessVS; PixelShader = decomp2; RenderTarget = texBloomCascade2; }
    pass { VertexShader = PostProcessVS; PixelShader = decomp3; RenderTarget = texBloomCascade3; }
    pass { VertexShader = PostProcessVS; PixelShader = decomp4; RenderTarget = texBloomCascade4; }
    pass { VertexShader = PostProcessVS; PixelShader = decomp5; RenderTarget = texBloomCascade5; }
    pass { VertexShader = PostProcessVS; PixelShader = decomp6; RenderTarget = texBloomCascade6; }
    // <>
    pass { VertexShader = PostProcessVS; PixelShader = recomp5; RenderTarget = texBloomCascade5; BlendEnable = true; BlendOp = ADD; SrcBlend = ONE; DestBlend = ONE; }
    pass { VertexShader = PostProcessVS; PixelShader = recomp4; RenderTarget = texBloomCascade4; BlendEnable = true; BlendOp = ADD; SrcBlend = ONE; DestBlend = ONE; }
    pass { VertexShader = PostProcessVS; PixelShader = recomp3; RenderTarget = texBloomCascade3; BlendEnable = true; BlendOp = ADD; SrcBlend = ONE; DestBlend = ONE; }
    pass { VertexShader = PostProcessVS; PixelShader = recomp2; RenderTarget = texBloomCascade2; BlendEnable = true; BlendOp = ADD; SrcBlend = ONE; DestBlend = ONE; }
    pass { VertexShader = PostProcessVS; PixelShader = recomp1; RenderTarget = texBloomCascade1; BlendEnable = true; BlendOp = ADD; SrcBlend = ONE; DestBlend = ONE; }
    pass { VertexShader = PostProcessVS; PixelShader = recomp0; RenderTarget = texBloomCascade0; BlendEnable = true; BlendOp = ADD; SrcBlend = ONE; DestBlend = ONE; }
    // <>
    pass { VertexShader = PostProcessVS; PixelShader = bandfetch; RenderTarget = texBloomFetch; }
    // <>
    pass { VertexShader = PostProcessVS; PixelShader = main; }
}