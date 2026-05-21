/*
   UNiT - Shader Library for ReShade.
   Analytical Ground-Truth Ambient Occlusion with Visibility Bitmask.
   More about it here: https://ar5iv.labs.arxiv.org/html/2301.11376
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform float WORLD_RADIUS
<
    ui_label = "Sampling Radius";
    ui_type = "slider";
    ui_min = 0.001f;
    ui_max = 1.0f;
> = 0.6f;

uniform float FADE_DISTANCE
<
    ui_label = "Fade Distance";
    ui_type = "slider";
    ui_min = 0.01f;
    ui_max = 1.0f;
> = 0.3;

uniform float Q
<
    ui_label = "Q (Learing Rate)";
    ui_tooltip = "Determines the filtering learning rate based on the current signal";
    ui_type = "slider";
    ui_min = 1.0;
    ui_max = 30.0;
    ui_step = 0.01;
    ui_category = "Denoising";
	ui_category_closed = true;
> = 10.0;

uniform float P
<
    ui_label = "P (Process Noise)";
    ui_tooltip = "Determines the filter confidence in signal constancy.";
    ui_type = "slider";
    ui_min = 0.0;
    ui_max = 0.01;
    ui_category = "Denoising";
    ui_category_closed = true;
> = 0.001;

uniform bool DEBUG
<
    ui_label = "Show Raw AO";
    ui_type = "radio";
> = false;

uniform int DEPTH_MODE
<
    ui_type = "combo";
    ui_items = "Default\0Inverse\0";
    ui_label = "Game Depth Mode";
    ui_tooltip = "Renders can use alternative depth mode. Turn inverse mode if shader don't work properly.";
    ui_category = "Scene Setup";
    ui_category_closed = true;
> = 0;

uniform int isRGB
<
    ui_type = "combo";
    ui_items = "sRGB\0Linear\0";
    ui_label = "Render Mode";
    ui_category = "Scene Setup";
    ui_category_closed = true;
> = 0;

/*=============================================================================
/   Preprocessing Definitions
/============================================================================*/
#include "ReShade.fxh"

#ifndef VBAO_SLICE_NUM
    #define VBAO_SLICE_NUM 4
#endif 

// math..
#define H_PI	1.570796326794
#define PI     	3.141592653589
#define TAU     6.283185307179

uniform int framecounter 
< 
    source = "framecount"; 
>;

texture texGBuffer 
{
    Width = BUFFER_WIDTH; 
    Height = BUFFER_HEIGHT; 
    Format = RGBA32F;
};

sampler sGBuffer 
{
    Texture = texGBuffer;
};

texture texRawOcclusion
{
    Width = BUFFER_WIDTH;   
    Height = BUFFER_HEIGHT;   
    Format = R8;
};

sampler sRawOcclusion
{
    Texture = texRawOcclusion;
};

storage stRawOcclusion
{ 
    Texture = texRawOcclusion; 
};

texture texGatheredOcclusion
{
    Width = BUFFER_WIDTH; 
    Height = BUFFER_HEIGHT; 
    Format = RG16F;
};

sampler sGatheredOcclusion
{
    Texture = texGatheredOcclusion;
};

texture texPackedHistory
{
    Width = BUFFER_WIDTH; 
    Height = BUFFER_HEIGHT; 
    Format = RG16F;
};

sampler sPackedHistory
{
    Texture = texPackedHistory;
};

texture texMotionVectors < pooled = false; > 
{ 
    Width = BUFFER_WIDTH; 
    Height = BUFFER_HEIGHT; 
    Format = RG16F; 
};

sampler SamplerMotionVectors 
{ 
    Texture = texMotionVectors; 
    AddressU = CLAMP; 
    AddressV = CLAMP; 
    MipFilter = POINT; 
    MinFilter = POINT; 
    MagFilter = POINT; 
};

/*=============================================================================
/   Global Helper Functions
/============================================================================*/
#define roundup(n, d) (int(n + d - 1) / int(d)) // ceil?

float pow2(float x)
{
    return x * x;
}

float2 facos(const float2 x) 
{
    float2 v = saturate(abs(x));
    float2 a = sqrt(1.0 - v) * (-0.16882 * v + 1.56734);
    return x > 0.0 ? a : PI - a;
}

float linear_to_z_plane(float depth) 
{
    return depth * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
}

float get_ndc_depth(float2 uv) 
{
    return !DEPTH_MODE ? linear_to_z_plane(ReShade::GetLinearizedDepth(uv)) : 1.0 / linear_to_z_plane(ReShade::GetLinearizedDepth(uv) + 1e-6);
}

float get_lin_depth(float2 uv)
{
    return ReShade::GetLinearizedDepth(uv);
}

float2 get_velocity_vector(float2 texcoord)
{
    return tex2D(SamplerMotionVectors, texcoord).rg;
}

float3 ndc_to_view(float2 uv, float z)
{ 
    float y = tan(radians(70) * 0.5);
    float x = y * BUFFER_ASPECT_RATIO;
    return float3(uv * 2.0 * float2(x, y) - float2(x, y), 1) * z;
}

float3 ndc_to_view(in float2 uv)
{
    return ndc_to_view(uv, get_ndc_depth(uv));
}

float2 view_to_uv(float3 pos)
{ 
    float y = tan(radians(70) * 0.5);
    float x = y * BUFFER_ASPECT_RATIO;
    float2 scale = 0.5 / float2(x, y);
    return (pos.xy / pos.z) * scale + 0.5;
}

float3 construct_normal(float2 uv)
{
    float2 o_h = float2(BUFFER_RCP_WIDTH, 0);
    float2 o_v = float2(0, BUFFER_RCP_HEIGHT);

    float3 center = ndc_to_view(uv);

    float4 h;
    h.x = ndc_to_view(uv - o_h).z;
    h.y = ndc_to_view(uv + o_h).z;
    h.z = ndc_to_view(uv - o_h * 2.0).z;
    h.w = ndc_to_view(uv + o_h * 2.0).z;
    float2 he = abs(h.xy * h.zw * rcp(2.0 * h.zw - h.xy) - center.z);

    float4 v;
    v.x = ndc_to_view(uv - o_v).z;
    v.y = ndc_to_view(uv + o_v).z;
    v.z = ndc_to_view(uv - o_v * 2.0).z;
    v.w = ndc_to_view(uv + o_v * 2.0).z;
    float2 ve = abs(v.xy * v.zw * rcp(2.0 * v.zw - v.xy) - center.z);

    float3 v_v = ve.x > ve.y ? ndc_to_view(uv + o_v) - center : center - ndc_to_view(uv - o_v);
    float3 h_v = he.x > he.y ? ndc_to_view(uv + o_h) - center : center - ndc_to_view(uv - o_h);

    return -normalize(cross(h_v, v_v));
}

void write_gbuffer(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target0)
{
    output = float4(construct_normal(texcoord), get_ndc_depth(texcoord));
}

/*=============================================================================
/   Workspace Helper Functions
/============================================================================*/
static const int permutation_table[8] = {0, 5, 2, 7, 1, 6, 3, 4}; // vogel > N=8

float2 octohedral_encode(float3 vec)
{
    float2 encoded = vec.xy / dot(abs(vec), 1.0);
    float2 sgn = vec.xy < 0.0 ? -1.0 : 1.0;
    return vec.z < 0 ? sgn - abs(encoded.yx) * sgn : encoded;
}

float3 cosinedir(float3 n, float r)
{
    float3 uu = normalize(cross(n, float3(0.0, 1.0, 1.0)));
    float3 vv = cross(uu, n);
    
    float rad = frac(r * 0.38196601125);

    float ra = sqrt(rad);
    float rx = ra * cos(TAU * r); 
    float ry = ra * sin(TAU * r);
    float rz = sqrt(1.0 - rad);
    float3 dir = normalize(rx*uu + ry*vv + rz*n);

    float ar = cos(PI * r) * 0.5 + 0.5;
    float w = rsqrt(ar * TAU); // importance-weighting

    return float3(octohedral_encode(dir) * w, 0.0);
}

float roberts1(float2 p)
{
    return frac(0.5 + dot(p, float2(0.569840290998, 0.7548776662467)));
}

float seedtemp(float2 pos)
{
    return roberts1(pos + permutation_table[framecounter % 8] * 6.2);
}

float sampleDiffuseIsotropic(sampler2D s, float2 uv, float2 xy)
{      
    float2 pattern = float2(-0.4726, -1.2742); // erf⁻¹(2u-1)*√2: u₁ = 1/π, u₂ = u₁²

    float a = tex2Dlod(s, float4(uv + float2( pattern.x,  pattern.y) * xy, 0, 0));
    float b = tex2Dlod(s, float4(uv + float2(-pattern.x, -pattern.y) * xy, 0, 0));
    float c = tex2Dlod(s, float4(uv + float2(-pattern.y,  pattern.x) * xy, 0, 0));
    float d = tex2Dlod(s, float4(uv + float2( pattern.y, -pattern.x) * xy, 0, 0));
    float e = tex2Dlod(s, float4(uv, 0, 0));

    return (a + b + c + d + e) / 5.0;  
}

float sampleDiffuseAnisotropic(sampler2D s, float2 uv, float2 xy, float sigma = 16.0)
{
    float2 tap = float2(0.0, 1.5);

    float a = tex2D(s, uv + tap.xy * xy);
    float b = tex2D(s, uv - tap.xy * xy);
    float c = tex2D(s, uv - tap.xx * xy);
    float d = tex2D(s, uv + tap.yx * xy);
    float e = tex2D(s, uv - tap.yx * xy);

    // exp(-|∇u|² / K²)
    float wa = exp(-pow2(abs(a-c) * sigma));
    float wb = exp(-pow2(abs(b-c) * sigma));
    float wd = exp(-pow2(abs(d-c) * sigma));
    float we = exp(-pow2(abs(e-c) * sigma));

    return c + 0.5 * (wa*(a-c) + wb*(b-c) + wd*(d-c) + we*(e-c));
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
static const uint sector_count = 32u;

float integrate_bitfield(in float2 h)
{
    uint bitfield_mask = 0xffffffffu; // https://issues.angleproject.org/issues/353039526

    uint2 h_id = uint2(round(h * sector_count));

    uint h_min = h_id.x < sector_count ? bitfield_mask << h_id.x : 0u;
    uint h_max = h_id.y != 0u ? bitfield_mask >> (sector_count - h_id.y) : 0u;

    uint occluded_bitfield = countbits(h_min & h_max); 
    
    return float(sector_count - occluded_bitfield) / float(sector_count);      
}

float tangent_horizon(float d_sq, float cos_h)
{
    float angle = 360.0 / float(sector_count);
    float maxcosine = cos(radians(angle));

    // the angle cannot be greater than the actual radius of the horizon coverage
    maxcosine = min(maxcosine, WORLD_RADIUS);

	float theta_sq = (1.0 - maxcosine) * (1.0 - maxcosine);
	float cos_phi = 4.0 * cos_h * (1.0 - cos_h);

    float tangent = (theta_sq * d_sq) / abs(cos_phi);

    return asin(tangent / PI);
}

[numthreads(24, 24, 1)] void write_occlusion(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= BUFFER_WIDTH || id.y >= BUFFER_HEIGHT) return;

    float2 uv = float2(id.xy + 0.5) * BUFFER_PIXEL_SIZE;
    int2 vpos = int2(id.xy + 0.5);

    float3 view_pos = ndc_to_view(uv);
    float3 view_dir = normalize(-view_pos);
    float3 normal = tex2Dfetch(sGBuffer, vpos.xy, 0).xyz;

    float world_radius = pow2(WORLD_RADIUS) * BUFFER_WIDTH;
    float pixel_radius = log2(view_pos.z + 1.0);

    float radius = world_radius / pixel_radius;
    float step_size = max(VBAO_SLICE_NUM, radius) / VBAO_SLICE_NUM;

	float2 h = -1.0;

    float random = seedtemp(float2(vpos.xy));
    float3 slice_dir = cosinedir(normal, random);

	float step_length = step_size;

	[loop]
	for (int slice = 0; slice < VBAO_SLICE_NUM; ++slice)
	{
        float increment = step_length * random;
	    float2 coord = slice_dir.xy * increment * BUFFER_PIXEL_SIZE;
	    float4 ruv = float4(uv + coord, uv - coord);

        if (any(saturate(ruv) != ruv))
        {
		    break;
   	    }

	    for (int pairs = 0; pairs < 2; ++pairs)
	    {
			float3 sample_pos = ndc_to_view(pairs == 0 ? ruv.xy : ruv.zw);
			float3 sample_center = sample_pos - view_pos;

			float dist_sqr = dot(sample_center, sample_center);
			float cos_horizon = rsqrt(dist_sqr) * dot(sample_center, view_dir);
            float max_horizon = cos_horizon - tangent_horizon(dist_sqr, cos_horizon);

			h[pairs] = max(h[pairs], max_horizon);
	    }

	    step_length += step_size;
	}

	h = facos(h);

	float3 tangent = normalize(cross(slice_dir, view_dir));
	float3 bitangent = cross(view_dir, tangent);
	float3 plane_normal = normal - tangent * dot(normal, tangent);
	float plane_distance = rsqrt(dot(plane_normal, plane_normal));

	float cos_n = dot(plane_normal, bitangent) * plane_distance;
	float gamma = facos(cos_n) - H_PI;
	float cos_gamma = dot(plane_normal, view_dir) * plane_distance;

	float3 ortho_dir = slice_dir - dot(slice_dir, view_dir) * view_dir;
    float theta_sign = sign(dot(ortho_dir, plane_normal)) * facos(cos_gamma);

    // clamp horizon into hemisphere
	h.x = gamma + max(-h.x - gamma, -H_PI);
	h.y = gamma + min( h.y - gamma,  H_PI);

    // TODO: In this implementation, I use one SPP for optimization. If multiple samples are used (in future maybe) necessary to flip the horizon on second sample
    // h = id > 0 ? h.yx : h.xy;

    // [0, 1]
	h = saturate((h + theta_sign + H_PI) / PI);
    
    float visibility = saturate(integrate_bitfield(h) * H_PI);

    tex2Dstore(stRawOcclusion, id.xy, float4(1.0 - visibility, 1.0, 1.0, 1.0));
}

void variance_online(inout float2 value, float2 uv)
{
	float2 uv_p = uv + tex2D(SamplerMotionVectors, uv).rg;
	float2 moments = tex2D(sPackedHistory, uv_p);

    bool isinscreen = all(uv_p > 0 && uv_p < BUFFER_SCREEN_SIZE);

    float signal = value.x;
    float prevsignal = moments.x;

	float pred = signal - prevsignal;

	if (!isinscreen) pred = 1.0;

	float dev = abs(pred);
	float cov_sq = (pred - moments.y) * (pred - moments.y);
    float dev_sq = dev * dev;

    float K = saturate((dev_sq * Q) + rsqrt(cov_sq) * P);
	
	value.x = lerp(prevsignal, signal, K);
	value.y = pred;
}

void guided_filtering(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float2 output : SV_Target)
{
    float2 signal = float2(sampleDiffuseIsotropic(sRawOcclusion, texcoord, BUFFER_PIXEL_SIZE), 1.0);  
    variance_online(signal, texcoord);  
    output = signal;
}

void pack_history(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float2 data : SV_Target)
{
    data = tex2D(sGatheredOcclusion, texcoord).xy;
}

float get_fade_factor(float2 uv)
{
    return saturate(saturate(length(ndc_to_view(uv)) / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE) / FADE_DISTANCE);
}

void main(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float3 output : SV_Target)
{
    float3 color = tex2Dfetch(ReShade::BackBuffer, vpos.xy, 0).rgb;
    float occlusion = sampleDiffuseAnisotropic(sGatheredOcclusion, uv, BUFFER_PIXEL_SIZE);

    float fadefactor = get_fade_factor(uv);
    occlusion = lerp(occlusion, 1.0, fadefactor);

	if (isRGB) color *= color;
    color = color / (1.0 - color + rcp(255.0));
    color *= occlusion;
    color = color / (1.0 + color);
    if (isRGB) color = sqrt(color);

    output = DEBUG ? occlusion.xxx : color;
}

/*=============================================================================
/   Technique Definition
/============================================================================*/
technique UNiT_GTVBAO < 
ui_label = "UNiT: GT-VBAO";
ui_tooltip = "									UNiT: A-GTVBAO \n\n" "________________________________________________________________________________________\n\n" "Screen-space ambient occlusion tech that operates on the analytical integration through\n" "horizon angle by using a visibility bit mask which gets a precise object shadowing.\n\n" " - Developed by RG2PS - "; >
{
    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = write_gbuffer;
	    RenderTarget = texGBuffer;
    }

    pass
    {
        ComputeShader = write_occlusion<24, 24>;
        DispatchSizeX = roundup(BUFFER_WIDTH, 24);
        DispatchSizeY = roundup(BUFFER_HEIGHT, 24);
    }

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = guided_filtering;
	    RenderTarget = texGatheredOcclusion;
    }

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = pack_history;
	    RenderTarget = texPackedHistory;
    }

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = main;
    }
}