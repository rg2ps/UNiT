/*
   UNiT - Shader Library for ReShade.
   Stochastic Filmic Grain via Monte-Carlo Integration
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform int _GrayFilm
<
    ui_type = "combo";
    ui_items = "Color\0Monochrome\0";
    ui_label = "Film Mode";
> = 1;

uniform bool _Animate
<
    ui_label = "Animate Grain";
    ui_type = "radio";
> = false;

uniform float A
<
    ui_type = "slider";
    ui_min = 0.01; ui_max = 1.0;
    ui_label = "Grain Strength";
    ui_category = "Grain Processing";
> = 0.6;

uniform float _Size
<
    ui_type = "slider";
    ui_min = 0.1; ui_max = 1.0;
    ui_label = "Grain Size";
    ui_category = "Grain Processing";
> = 0.5;

/*=============================================================================
/   Buffer Samplers Definition
/============================================================================*/
#include "ReShade.fxh" 

uniform int frameCount < source = "framecount"; >;

#define MAX_GREY_SCALE 256

texture texLambdaLUT 
{
    Width = MAX_GREY_SCALE; 
    Height = 1;
    Format = RG16F;
};

sampler sLambdaLUT 
{ 
    Texture = texLambdaLUT;
};

texture texPoissonResult
{
    Width = BUFFER_WIDTH; 
    Height = BUFFER_HEIGHT; 
    Format = RGBA16F;
};

sampler sPoissonResult
{ 
    Texture = texPoissonResult; 
};

/*=============================================================================
/   Global Helper Functions
/============================================================================*/
float3 to_linear(float3 x)
{
    return x * x * sign(x);
}

float3 from_linear(float3 x)
{
    return sqrt(abs(x)) * sign(x);
}

float gray3(float3 x)
{
    return from_linear(dot(to_linear(x), float3(0.2126729, 0.7151522, 0.072175)));
}

float erfinv(float x) 
{
    float tt1, tt2, lnx, sgn;
    sgn = (x < 0.0f) ? -1.0f : 1.0f;
    x = (1.0f - x) * (1.0f + x);
    lnx = log(x);
    tt1 = 2.0f / (3.14159265359f * 0.147f) + 0.5f * lnx;
    tt2 = 1.0f / (0.147f) * lnx;
    return sgn * sqrt(-tt1 + sqrt(tt1 * tt1 - tt2));
}

/*=============================================================================
/   Workspace Helper Functions
/============================================================================*/
uint hash32(uint x)
{
    x ^= (x >> 16) | 1u; // don't hash zero, at least one bit
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

uint hash32(uint2 x)
{
    return hash32(hash32(x.y) + x.x);
}

float rand_uniform_0_1(uint2 x) 
{
    return float(hash32(x)) * exp2(-32.0);
}

uint signbit(uint x)
{
    return 31 - firstbithigh(x | 1);
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
float num_grains()
{
    return min(rcp(A), sqrt(MAX_GREY_SCALE));
}

float num_grains_inv()
{
    return saturate(A + 0.5) * num_grains();
}

void poisson_lut(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float2 output : SV_Target)
{
    const int max_level = MAX_GREY_SCALE - 1;
    int level = (int)(texcoord.x * max_level);
    float exposure = (float)level / max_level; // ev in [0, 1]

    float beta = 1.0 / (num_grains() * num_grains());
    float lambda = -log(1.0 - to_linear(exposure)) * num_grains();

    // neg-binomial <-> poisson
	float Q = lambda / (sqrt(lambda) * (1.0 - beta) + beta);

    output = float2(Q, exp(-Q));  
}

float gen_poisson(float2 lambda, uint seed, uint mip)
{
	int k = 0;              // num of trials
    int strats = 1;         // num of colors

    float p = 1.0;
    float pdf = lambda.y;

	[loop]
	do 
    {
        strats += signbit(mip * seed);
        
	    k++;
	    p *= rand_uniform_0_1(k * (seed + strats));
	    
	    pdf *= lambda.x / float(k);
    }
    while (p > lambda.y); 
    
    return float(k - 1) - pdf * 0.5;
}

void monte_carlo(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    uint2 pos = uint2(vpos.xy);
    
    uint seed = hash32(pos);
    if (_Animate) seed += reversebits(frameCount);

    float3 color = tex2Dfetch(ReShade::BackBuffer, pos, 0).rgb; 
    float gray_color = gray3(color);

    float3 poisson = 0.0;

    if (!_GrayFilm)
    {
        poisson.x = gen_poisson(tex2Dlod(sLambdaLUT, float4(color.x, 0, 0, 0)).xy, seed, 0);
        poisson.y = gen_poisson(tex2Dlod(sLambdaLUT, float4(color.y, 0, 0, 0)).xy, seed, 1);
        poisson.z = gen_poisson(tex2Dlod(sLambdaLUT, float4(color.z, 0, 0, 0)).xy, seed, 2);
    }
    else
    {
        // to simulate b/w film, use two layers of emulsion, ensures they are uncorrelated with each other.
        poisson += gen_poisson(tex2Dlod(sLambdaLUT, float4(gray_color, 0, 0, 0)).xy, seed + 0, 3) * 0.5;
        poisson += gen_poisson(tex2Dlod(sLambdaLUT, float4(gray_color, 0, 0, 0)).xy, seed + 1, 4) * 0.5;
    }
    
    // back to sdr
    output = float4(poisson / num_grains_inv(), 1.0);
}

void main(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float3 output : SV_Target)
{
    float3 color = tex2Dfetch(ReShade::BackBuffer, vpos.xy, 0).rgb; 
    if (_GrayFilm) color = gray3(color);

    float3 sum = 0.0;
    float total = 0.0;

    float2 width = float2(1.0, 0.5 * _Size);

    [unroll]for(int i = -1; i <= 1; i++)
    [unroll]for(int j = -1; j <= 1; j++)
    {
        uint2 pos = uint2(vpos.xy) + uint2(i, j);
	    float3 poisson = tex2Dfetch(sPoissonResult, pos, 0).rgb;

        float seed = rand_uniform_0_1(pos);
        float ndf = erfinv(seed * 2.0 - 1.0);

	    float sigma = rsqrt(2.0 / num_grains());
        float cell_seed = ndf / (2.0 * sigma * sigma);

        float x = length(float2(i, j) + cell_seed);
        float weight = exp(-x * x);
        
	    weight *= width[abs(i)] * width[abs(j)];

	    sum += poisson * weight;
	    total += weight;
    }

    float3 color_poisson = from_linear(sum / total);
    
    output = lerp(color, color_poisson, 1.0 / num_grains());
}

/*=============================================================================
/   Technique Definition
/============================================================================*/
technique UNiTFilmGrain < 
ui_label = "UNiT: Film Grain";
ui_tooltip = "                      		UNiT: Stochastic Film Grain \n\n" "_______________________________________________________________________________________\n\n" "Physically based filmic grain model approach which operates on the monte-carlo poisson\n" "integration for pure realistic simulation of the photo-emulsion process.\n\n" " - Developed by RG2PS - "; >
{

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = poisson_lut;
	    RenderTarget = texLambdaLUT;
    }

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = monte_carlo;
	    RenderTarget = texPoissonResult;
    }

    pass
    {
    	VertexShader = PostProcessVS;
	    PixelShader = main;
    }
}
