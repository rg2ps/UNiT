/*
   UNiT - Shader Library for ReShade.
   Stochastic Filmic Grain via Quasi-Continuent Poisson Integration
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform int _GrayFilm
<
    ui_type = "combo";
    ui_items = "Default\0Monochrome\0";
    ui_label = "Film Mode";
> = 0;

uniform bool _Animate
<
    ui_label = "Animate Grain";
    ui_type = "radio";
> = false;

uniform float _Amount
<
    ui_type = "slider";
    ui_min = 0.01; ui_max = 1.0;
    ui_label = "Grain Strength";
    ui_category = "Grain Processing";
> = 0.5;

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
float gray3(float3 x)
{
    return dot(x, float3(0.2126729, 0.7151522, 0.072175));
}

float3 to_linear(float3 x)
{
    return x * x * sign(x);
}

float3 from_linear(float3 x)
{
    return sqrt(abs(x)) * sign(x);
}

float erfinv(float x) 
{
    float lx = log((1.0f - x) * (1.0f + x));
    float t1 = 2.0f / (3.14159265359f * 0.147f) + 0.5f * lx;
    float t2 = 1.0f / (0.147f) * lx;
    return sqrt(-t1 + sqrt(t1 * t1 - t2)) * sign(x);
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

uint lk_hash(uint x, uint n) 
{
    // owen-permutation hash from the paper
    x ^= x * 0x3d20adeau;
    x += n;
    x *= (n >> 16) | 1u;
    x ^= x * 0x05526c56u;
    x ^= x * 0x53a22864u;
    return x;
}

float rand_uniform_0_1(uint x, uint n) 
{
    return float(lk_hash(x, n)) * exp2(-32.0);
}

float rand_uniform_0_1(uint2 x) 
{
    return float(hash32(x)) * exp2(-32.0);
}

// Converts conditional intensity units to film developing value in [0, 1]
float develop_scale(float k = 8.0)
{
    return (exp2(_Amount * ((k / 2.0) - 1.0)) - exp2(-k * sqrt(_Amount))) / k;
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
void write_lut(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float2 output : SV_Target)
{
    const int max_level = MAX_GREY_SCALE - 1;
    int level = (int)(texcoord.x * max_level);
    float C_u = (float)level / max_level; // exposure in [0, 1]

    float Hc = _Amount; // grain density per pixel here
    float Hc2 = Hc * Hc;

    float lambda = min(max_level, -log(1.0 - to_linear(C_u)) * rsqrt(Hc));

    // Poisson <-> Neg-Binomial
	float Q = lambda / (sqrt(lambda) * (1.0 - Hc2) + Hc2);

    output = float2(Q, exp(-Q));  
}

// Stirling's approximation of: λ^k * e(-λ)/k!
float factorial(int k, float l)
{
    return exp(k * log(l) - l - (k * log(k) - k + 0.5 * log(6.283185307 * k)));
}

float gen_poisson(float2 lambda, uint seed, uint mip)
{
	int k = 0;      // num of trials
    int n = 1;	    // num of layers per trial

    float p = 1.0;

	[loop]
	do 
    {
        n += 31 - firstbithigh(mip * seed | 1); // some trick that models the random colored points clustering

	    k++;
	    p *= rand_uniform_0_1(k, seed + reversebits(n));
    } 
    while (p > lambda.y && k < 255);

    return float(k - 1) - factorial(k, lambda.x);
}

void monte_carlo(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    uint2 pos = uint2(vpos.xy);
    
    uint seed = hash32(pos);
    if (_Animate) seed += frameCount;

    float3 color = tex2Dfetch(ReShade::BackBuffer, pos, 0).rgb; 

    float3 poisson = 0.0;

    if (!_GrayFilm)
    {
        poisson.x = gen_poisson(tex2Dlod(sLambdaLUT, float4(color.x, 0, 0, 0)).xy, seed, 0);
        poisson.y = gen_poisson(tex2Dlod(sLambdaLUT, float4(color.y, 0, 0, 0)).xy, seed, 1);
        poisson.z = gen_poisson(tex2Dlod(sLambdaLUT, float4(color.z, 0, 0, 0)).xy, seed, 2);
    }
    else
    {
        float luma = gray3(color);
        poisson = gen_poisson(tex2Dlod(sLambdaLUT, float4(luma, 0, 0, 0)).xy, seed, 0).xxx;
    }

    output = float4(poisson * sqrt(_Amount), 1.0);
}

void main(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float3 output : SV_Target)
{
    float3 color = tex2Dfetch(ReShade::BackBuffer, vpos.xy, 0).rgb; 

    float3 sum = 0.0;
    float total = 0.0;

    float2 width = float2(1.0, 0.5 * _Size);

    [unroll]for(int i = -1; i <= 1; i++)
    [unroll]for(int j = -1; j <= 1; j++)
    {
        uint2 pos = uint2(vpos.xy) + uint2(i, j);
	    float3 poisson = tex2Dfetch(sPoissonResult, pos, 0).rgb;

        float seed = rand_uniform_0_1(pos);
        float gaussian_ndf = erfinv(seed * 2.0 - 1.0);

	    float sigma = rsqrt(2.0 * _Amount);
        float x = length(float2(i, j) + gaussian_ndf / (2.0 * sigma * sigma));
        float weight = exp(-x * x);
        
	    weight *= width[abs(i)] * width[abs(j)];

	    sum += poisson * weight;
	    total += weight;
    }

    float3 color_poisson = sum / total;
    
    color = to_linear(color);
    if (_GrayFilm) color = gray3(color);

    color = lerp(color, color_poisson, develop_scale());  

    color = from_linear(color);

    output = color;
}

/*=============================================================================
/   Technique Definition
/============================================================================*/
technique UNiTFilmGrain < 
ui_label = "UNiT: Film Grain";
ui_tooltip = "                      		UNiT: Stochastic Film Grain \n\n" "_____________________________________________________________________________________\n\n" "Physically based filmic grain model approach which operates on the continuum poisson\n" "integration for pure realistic simulation of the photo-emulsion process.\n\n" " - Developed by RG2PS - "; >
{

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = write_lut;
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
