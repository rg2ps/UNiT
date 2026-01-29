/*
   UNiT - Shader Library for ReShade.
   Stochastic Filmic Grain via Poisson Distribution.
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform int _Cm
<
    ui_type = "combo";
    ui_items = "Colored\0White\0";
    ui_label = "Grain Mode";
    ui_category = "Global";
> = 0;

uniform int _Fm
<
    ui_type = "combo";
    ui_items = "Default\0Monochrome\0";
    ui_label = "Film Mode";
    ui_category = "Global";
> = 0;

uniform bool _Animate
<
    ui_label = "Animate Grain";
    ui_category = "Global";
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
> = 0.85;

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

float3 from_hdr(float3 x)
{
    return saturate(1.1 * x * rsqrt(1 + x * x));
}

float3 to_hdr(float3 x)
{
   return min(16.0, 0.9090909 * x * rsqrt((1 - x * x) + 0.003921));
}

/*=============================================================================
/   Workspace Helper Functions
/============================================================================*/
uint lowbias32(uint x)
{
    x ^= (x >> 16) | 1u;
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

uint3 to_next3u(uint h)
{
    // Optimized for quasi-random sequences, pure different word at each bit change.
    return h * uint3(0xdb01bd51u, 0xdaef3c2cu, 0x75aeb75bu);
}

float qmc(uint x, uint n) 
{
    return float(lk_hash(reversebits(x), reversebits(n))) * exp2(-32.0);
}

uint lowbias_rng32(uint2 x)
{
    return lowbias32(lowbias32(x.y) + x.x);
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
void write_lut(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float2 output : SV_Target)
{
    const int max_level = MAX_GREY_SCALE - 1;
    int level = (int)(texcoord.x * max_level);
    float C_u = (float)level / max_level;

    float Hc = _Amount; // grains density per pixel here
    float Hc2 = Hc * Hc;

    float x = min((float)max_level, -log(1.0 - C_u * C_u * sign(C_u)) * rcp(Hc));

    // For high λ, the grains should a clusterize approximately by √λ, that is approach a normal distribution
    // This is necessary to correctly count n-th number of points (batches) at a one event
    float lambda = x / (sqrt(x) * (1.0 - Hc2) + Hc2);

    output = float2(lambda, exp(-lambda));  
}

void poisson_process(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    float3 color = tex2Dfetch(ReShade::BackBuffer, vpos.xy, 0).rgb; 

    uint2 pos = uint2(vpos.xy);
    
    if (_Animate) 
	{
		pos |= ~frameCount * pos.x * pos.y;
    }

    float3 event = 0.0;

    [unroll]
    for (int c = 0; c < 3; ++c) 
    {
        float exposure = color[c];

	    float2 l = tex2Dlod(sLambdaLUT, float4(float2(exposure, 0.5), 0, 0)).xy;
 	    float p = 1.0;
	    int k = 0;
	    
        uint3 r = lowbias_rng32(pos).xxx;

        if (!(_Fm || _Cm))
        {
            r = to_next3u(r);
        }

	    [loop]
	    do {
	        k++;
	        p *= qmc(uint(k), r[c]);
	    } while (p >= l.y && k <= 255);

		// The less clusters, the more counter integer..
		float a3 = _Amount * _Amount * _Amount * 0.78539;
		float cut = 1.0 - a3;
        float k_fract = max(0, p % sqrt(l.x)) *  cut;
	    event[c] = float(k-1) - k_fract;
    }

    float3 color_poisson = event * _Amount;

    output = float4(!_Fm ? color_poisson : gray3(color_poisson), 1);
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
        
        float weight = 1.0;

        float x = length(float2(i, j));
	    float sigma = rsqrt(_Amount) * 0.785398;
	    float mu = 0.3465735903f - (sigma * sigma) * 0.5f;
	    float u = (log(x) - mu) / sigma;
        if (x != 0) weight = exp(-0.5 * u * u) / (x * sigma * 2.50662827f);
        
	    weight *= width[abs(i)] * width[abs(j)];

	    sum += poisson * weight;
	    total += weight;
    }

    float3 color_poisson = sum / total;
    
    color = to_hdr(to_linear(color));

    if (_Fm) color = gray3(color);

    color = lerp(color, color_poisson, _Amount);  
    
    color = from_linear(from_hdr(color));

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
