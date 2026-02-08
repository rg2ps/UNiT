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

float3 from_linear(float3 x)
{
    return lerp(12.92*x, 1.055 * pow(x, 0.4166666666666667) - 0.055, step(0.0031308, x));
}

float3 to_linear(float3 x)
{
    return lerp(x / 12.92, pow((x + 0.055)/(1.055), 2.4), step(0.04045, x));
}

float to_linear(float x)
{
    return x * x * sign(x);
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

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
void write_lut(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float2 output : SV_Target)
{
    const int max_level = MAX_GREY_SCALE - 1;
    int level = (int)(texcoord.x * max_level);
    float C_u = (float)level / max_level;

 	// Grains density per pixel here
    float Hc = _Amount;
    float Hc2 = Hc * Hc;

    float lambda = min(max_level, -log(1.0 - to_linear(C_u)) * rsqrt(Hc));

    // Poisson <-> Neg-Binomial
	float Q = lambda / (sqrt(lambda) * (1.0 - Hc2) + Hc2);

    output = float2(Q, exp(-Q));  
}

// Stirling's approximation of: √2 × λ^k * exp(-λ)/k!
float factorial(int k, float l)
{
    return exp(k * log(l) - l - (k * log(k) - k + 0.5 * log(6.283185307 * k))) * 1.4142135623;
}

// Converts conditional intensity units to film developing value in [0, 1]
float develop(float k = 8.0)
{
    return (exp2(_Amount * ((k / 2.0) - 1.0)) - exp2(-k * sqrt(_Amount))) / k;
}

void poisson_process(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    uint2 pos = uint2(vpos.xy);
    
    float3 color = tex2Dfetch(ReShade::BackBuffer, pos, 0).rgb; 
    if (_GrayFilm) color = gray3(color);
    
    uint seed = hash32(pos);
    if (_Animate) seed += frameCount;

    float3 event = 0.0;

    for (int i = 0; i < (_GrayFilm ? 1 : 3); ++i) 
    { 	
	    float2 l = tex2Dlod(sLambdaLUT, float4(color[i], 0, 0, 0)).xy;   
	    
	    int trial = 0;  // num of trials
        int layer = 1;	// num of layers per trial
 	    float p = 1.0;  // uniform prod	

	    [loop]
	    do 
        {
        	if (!_GrayFilm) layer += (i + 1) * 2;
        	
	        trial++;
	        p *= rand_uniform_0_1(trial, seed + reversebits(layer));
	       
	    } 
        while (p > l.y && trial < 255);

        float continuum = factorial(trial, l.x);
        
        if (!_GrayFilm)
	    	event[i] = float(trial - 1) - continuum;
	    else
	    	event.xyz = float(trial - 1) - continuum;
    }

    output = float4(event * sqrt(_Amount), 1);
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

    color = lerp(color, color_poisson, develop());  

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
	    PixelShader = poisson_process;
	    RenderTarget = texPoissonResult;
    }

    pass
    {
    	VertexShader = PostProcessVS;
	    PixelShader = main;
    }
}
