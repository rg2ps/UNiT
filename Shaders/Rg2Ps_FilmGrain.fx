/*
   UNiT - Shader Library for ReShade.
   Stochastic Filmic Grain via Poisson Distribution.
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform int _Fm
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

float3 to_linear_fast(float3 x)
{
    return x * x * sign(x);
}

/*=============================================================================
/   Workspace Helper Functions
/============================================================================*/
uint lowbias32(uint x)
{
    x ^= (x >> 16) | 1u; // don't hash zero, at least one bit
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
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

uint lowbias32(uint2 x)
{
    return lowbias32(lowbias32(x.y) + x.x);
}

float rand_uniform(uint x, uint n) 
{
    return float(lk_hash(x, n)) * exp2(-32.0);
}

float continuum(float probability, float lambda)
{
	// approximation of the continuous poisson quantities
    float error = (1.0 + rsqrt(MAX_GREY_SCALE)) - sqrt(_Amount);
    float part = min(error, probability % sqrt(lambda));
    return max(0.0, part);
}

float erfinv(float x) 
{
    float lnx = log((1.0f - x) * (1.0f + x));
    float tt1 = 2.0f / (3.14159265359f * 0.147f) + 0.5f * lnx;
    float tt2 = 1.0f / (0.147f) * lnx;
    float ndf = sqrt(-tt1 + sqrt(tt1 * tt1 - tt2)) * sign(x);
    return clamp(ndf * sqrt(2.0), -0.99, 0.99);
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

    float lambda = min((float)max_level, -log(1.0 - to_linear_fast(C_u)) * rsqrt(Hc));

    // Poisson <-> Neg-Binomial
	float Q = lambda / (sqrt(lambda) * (1.0 - Hc2) + Hc2);

    output = float2(Q, exp(-Q));  
}

void poisson_process(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    uint2 pos = uint2(vpos.xy);
    
    float3 color = tex2Dfetch(ReShade::BackBuffer, pos, 0).rgb; 
    if (_Fm) color = gray3(color);
    
    uint seed = lowbias32(pos);
    if (_Animate) seed += frameCount;

    float3 event = 0.0;

    for (int j = 0; j < (_Fm ? 1 : 3); ++j) 
    { 	
	    float2 l = tex2Dlod(sLambdaLUT, float4(color[j], 0, 0, 0)).xy ;   
	    
	    int trial = 0; // num of trials
 	   float p = 1.0; // uniform prod	
 	   uint L = 1;	// num of layers per trial

	    [loop]
	    do 
        {
        	L += (j + 1) * 2;
	        trial++;
	        p *= rand_uniform(trial, seed + (_Fm ? 0 : reversebits(L)));
	       
	    } 
        while (p > l.y && trial < 255);
        
        if (!_Fm)
	    	event[j] = float(trial - 1) - continuum(p, l.x);
	    else
	    	event = float(trial - 1) - continuum(p, l.x);
    }

    // back to sdr value
    float3 poisson = min(1.25331414, event * sqrt(_Amount));
    
    output = float4(poisson, 1);
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
        float seed = float(lowbias32(pos)) * exp2(-32.0);;
        float ndf = erfinv(seed);

	    float3 poisson = tex2Dfetch(sPoissonResult, pos, 0).rgb;

	    float sigma = rsqrt(2.0 * _Amount);
        float x = length(float2(i, j) + ndf / (2.0 * sigma * sigma));
	    float mu = 0.3465735903f - (sigma * sigma) * 0.5f;
	    float u = (log(x) - mu) / sigma;
        float weight = x > 0 ? exp(-0.5 * abs(u)) / (x * sigma * 2.50662827f) : 1;
        
	    weight *= width[abs(i)] * width[abs(j)];

	    sum += poisson * weight;
	    total += weight;
    }

    float3 color_poisson = sum / total;
    
    color = to_linear(color);
       if (_Fm) color = gray3(color);
    
    
    color = lerp(color, color_poisson, _Amount);  

    color = from_linear(color);

    output = color;
}

/*=============================================================================
/   Technique Definition
/============================================================================*/
technique UNiTFilmGrain < 
ui_label = "UNiT: Film Grain";
ui_tooltip = "                      		UNiT: Stochastic Film Grain \n\n" "________________________________________________________________________________________\n\n" "Physically based filmic grain model approach which operates on the poisson distribution\n" "for realistic simulation of the photo-emulsion process.\n\n" " - Developed by RG2PS - "; >
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
