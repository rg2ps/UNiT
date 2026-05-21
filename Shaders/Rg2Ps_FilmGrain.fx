/*
   UNiT - Shader Library for ReShade.
   Stochastic Film Grain Rendering via Monte-Carlo Integration
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform int FILM_MODE
<
    ui_type = "combo";
    ui_items = "Color\0Monochrome\0";
    ui_label = "Film Mode";
> = 0;

uniform bool ANIMATION
<
    ui_label = "Animate Grain";
    ui_type = "radio";
> = false;

uniform float GRAIN_AMOUNT
<
    ui_type = "slider";
    ui_min = 0.01; ui_max = 1.0;
    ui_label = "Grain Strength";
    ui_category = "Grain Processing";
> = 0.6;

uniform float GRAIN_SIZE
<
    ui_type = "slider";
    ui_min = 0.1; ui_max = 1.0;
    ui_label = "Grain Size";
    ui_category = "Grain Processing";
> = 0.35;

/*=============================================================================
/   Buffer Samplers Definition
/============================================================================*/
#include "ReShade.fxh" 

uniform int frameCount 
< 
    source = "framecount"; 
>;

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
float3 from_gamma(float3 x)
{
    return x * x * sign(x);
}

float3 to_gamma(float3 x)
{
    return sqrt(abs(x)) * sign(x);
}

float gray3(float3 x)
{
    return to_gamma(dot(from_gamma(x), float3(0.2126729, 0.7151522, 0.072175)));
}

/*=============================================================================
/   Workspace Helper Functions
/============================================================================*/
uint hash32(uint seed) 
{
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint hash32(uint2 x)
{
    return hash32(hash32(x.y) + x.x);
}

uint2 u_to_u2(uint x)
{
    return reversebits(uint2(x >> 16, x & 0xffff));
}

float rand_uniform_0_1(uint2 x) 
{
    return float(hash32(x)) * exp2(-32.0);
}

float2 rand_uniform_0_1(uint2 x, uint k) 
{
    return float2(u_to_u2(hash32(x))) * exp2(-32.0);
}

float2 boxmuller(float2 u) 
{
    float2 dir; 
	sincos(6.281853 * u.x, dir.y, dir.x);
    return dir * sqrt(-2.0 * log(1.0 - u.y));
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
float num_grains()
{
    return min(rcp(GRAIN_AMOUNT), sqrt(MAX_GREY_SCALE));
}

float num_grains_inv()
{
    return saturate(GRAIN_AMOUNT + 0.5) * num_grains();
}

void poisson_lut(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float2 output : SV_Target)
{
    const int max_level = MAX_GREY_SCALE - 1;
    int level = (int)(texcoord.x * max_level);
    float exposure = (float)level / max_level; // ev in [0, 1]

    float beta = 1.0 / (num_grains() * num_grains());
    float lambda = -log(1.0 - from_gamma(exposure)) * num_grains();

    // neg-binomial <-> poisson
	float q = lambda / (sqrt(lambda) * (1.0 - beta) + beta);

    output = float2(q, exp(-q));  
}

float gen_poisson(float2 lambda, uint seed, uint mip)
{
	int k = 0;

    float p = 1.0;
    float pdf = lambda.y;

	[loop]
	do 
    {
	    k++;

        float spp = rand_uniform_0_1(uint2(seed + k, mip));

	    p *= spp;
	    pdf *= lambda.x / float(k);
    }
    while (p > lambda.y); 
    
    return float(k - 1) - pdf * 0.5;
}

void monte_carlo(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    uint2 pos = uint2(vpos.xy);
    
    uint seed = hash32(pos);
    if (ANIMATION) seed += frameCount << 2;

    float3 color = saturate(tex2Dfetch(ReShade::BackBuffer, pos, 0).rgb); 
    float luma = gray3(color);

    float3 poisson = 0.0;

    if (!FILM_MODE)
    {
        poisson.x = gen_poisson(tex2Dlod(sLambdaLUT, float4(color.x, 0, 0, 0)).xy, seed, 0);
        poisson.y = gen_poisson(tex2Dlod(sLambdaLUT, float4(color.y, 0, 0, 0)).xy, seed, 1);
        poisson.z = gen_poisson(tex2Dlod(sLambdaLUT, float4(color.z, 0, 0, 0)).xy, seed, 2);
    }
    else
    {
        poisson += gen_poisson(tex2Dlod(sLambdaLUT, float4(luma, 0, 0, 0)).xy, seed, 0);
    }

    poisson /= num_grains_inv();
    
    // back to sdr
    output = float4(min(sqrt(2.0), poisson), 1.0);
}

void main(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float3 output : SV_Target)
{
    float3 color = tex2Dfetch(ReShade::BackBuffer, vpos.xy, 0).rgb; 
    if (FILM_MODE) color = gray3(color);

    float3 sum = 0.0;
    float total = 0.0;

    float2 gaussian = float2(1.0, GRAIN_SIZE / 2.0);

    [unroll]for(int i = -1; i <= 1; i++)
    [unroll]for(int j = -1; j <= 1; j++)
    {
        uint2 pos = uint2(vpos.xy) + uint2(i, j);
	    float3 poisson = tex2Dfetch(sPoissonResult, pos, 0).rgb;

        float2 seed = rand_uniform_0_1(pos, 0);
        float2 ndf = boxmuller(seed);
        float2 cellseed = ndf / num_grains();

        float2 x = float2(i, j) + cellseed;
        float weight = exp(-dot(x, x));
        
	    weight *= gaussian[abs(i)] * gaussian[abs(j)];

	    sum += poisson * weight;
	    total += weight;
    }

    float3 color_poisson = to_gamma(sum / total);
    
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
