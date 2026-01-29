/*
   UNiT - Shader Library for ReShade.
   Analytical Ground-Truth Ambient Occlusion with Visibility Bitmask.
   More about it here: https://ar5iv.labs.arxiv.org/html/2301.11376
   
   Written for ReShade by RG2PS (c) 2026. Provided by EULA.
   Any file parts redistribution only with permission. All right reserved.
   Read the end-user license agreement to get more details.
*/

uniform float _Intensity
<
    ui_label = "Occlusion Strength";
    ui_type = "slider";
    ui_min = 0.5f;
    ui_max = 2.0f;
> = 1.0f;

uniform float _Radius
<
    ui_label = "Sampling Radius";
    ui_type = "slider";
    ui_min = 0.001f;
    ui_max = 1.0f;
> = 0.6f;

uniform float _FadeDist
<
    ui_label = "Fade Distance";
    ui_type = "slider";
    ui_min = 0.01f;
    ui_max = 1.0f;
> = 0.7;

uniform float _LearningRate
<
    ui_label = "Temporal Filtering";
    ui_type = "drag";
    ui_min = 0.1f;
    ui_max = 0.9f;
> = 0.5f;

uniform bool _Debug
<
    ui_label = "Enable Debug Mode";
    ui_type = "radio";
> = false;

uniform int _Zm
<
    ui_type = "combo";
    ui_items = "Default\0Inverse\0";
    ui_label = "Game Depth Mode";
    ui_tooltip = "Games can use alternative depth mode. Turn inverse mode if shader don't work properly.";
> = 0;

/*=============================================================================
/   Buffer Samplers Definition
/============================================================================*/
#include "ReShade.fxh"

#define H_PI	1.570796326794
#define PI     	3.141592653589
#define TAU     6.283185307179

#define MAX_SLICES_NUM  4
#define MAX_SAMPLES_NUM 2

uniform int frameCount < source = "framecount"; >;

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

texture texReferanceOcclusion
{
    Width = BUFFER_WIDTH; 
    Height = BUFFER_HEIGHT; 
    Format = RGBA16F;
};

sampler sReferanceOcclusion
{
    Texture = texReferanceOcclusion;
};

texture texPackedHistory
{
    Width = BUFFER_WIDTH; 
    Height = BUFFER_HEIGHT; 
    Format = RGBA16F;
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
float linear_to_z_plane(float depth) 
{
    return depth * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
}

float get_ndc_depth(float2 uv) 
{
    return !_Zm ? linear_to_z_plane(ReShade::GetLinearizedDepth(uv)) : 1.0 / linear_to_z_plane(ReShade::GetLinearizedDepth(uv) + 1e-6);
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
#define roundup(n, d) (int(n + d - 1) / int(d)) // ceil?

float2 facos(const float2 x) 
{
    float2 v = saturate(abs(x));
    float2 a = sqrt(1.0 - v) * (-0.16882 * v + 1.56734);
    return x > 0.0 ? a : PI - a;
}

float weyl_1d(float2 p)
{
    return frac(0.5 + dot(p, float2(0.569840290998, 0.7548776662467)));
}

float2 weyl_2d(float n)
{
    return frac(0.5 + n * float2(0.7548776662467, 0.569840290998));
}

float weyl_temporal(float2 pos)
{
    float shift;
    
    int f = frameCount % 8;
    int f2b = ((f&4)>>2) | ((f&2)) | ((f&1)<<2);

    shift = float(f2b) * 6.018;

    // TODO, clip to 16 least major bits = better TAA consistency, but I could be wrong
    shift = float((uint)shift & 0xffffu);

    return weyl_1d(pos + shift);
}

float2 erfinv(float2 x) 
{
    float2 tt1, tt2, lnx, sgn;
    sgn = (x < 0.0f) ? -1.0f : 1.0f;
    x = (1.0f - x) * (1.0f + x);
    lnx = log(x);
    tt1 = 2.0f / (3.14159265359f * 0.147f) + 0.5f * lnx;
    tt2 = 1.0f / (0.147f) * lnx;
    return (sgn * sqrt(-tt1 + sqrt(tt1 * tt1 - tt2)));
}

float get_fade_factor(float2 uv)
{
    return saturate(saturate(length(ndc_to_view(uv)) / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE) / _FadeDist);
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
[numthreads(24, 24, 1)] void write_occlusion(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= BUFFER_WIDTH || id.y >= BUFFER_HEIGHT) return;

    float2 uv = float2(id.xy + 0.5) * BUFFER_PIXEL_SIZE;
    int2 vpos = int2(id.xy + 0.5);

    float3 view_pos = ndc_to_view(uv) * 0.999;
    float3 view_dir = normalize(-view_pos);
    float3 normal = tex2Dfetch(sGBuffer, vpos.xy, 0).xyz;

    float world_radius = _Radius * _Radius * BUFFER_WIDTH;
    float relative_radius = log2(view_pos.z + 1e-6);
    float radius = world_radius / relative_radius;
    float step_size = max(MAX_SLICES_NUM, radius) / MAX_SLICES_NUM;

    float random = weyl_temporal(float2(vpos.xy));

    uint occluded_bitfield = 0u;
    float visibility = 0.0;

    [loop]
    for (int s_idx = 0; s_idx < MAX_SAMPLES_NUM; ++s_idx)
    {
        float2 r = float2(random, random + float(s_idx));

        float3 uu = normalize(cross(normal, float3(0.0, 1.0, 1.0)));
        float3 vv = cross(uu, normal);

        float3 v;
        sincos(TAU * r.x, v.y, v.x);
        v.xy *= sqrt(r.y);
        v.z = sqrt(1.0 - r.y);

        float3 uvec = normalize(v.x*uu + v.y*vv + v.z*normal);
        float3 slice_dir = float3(uvec.xy / dot(abs(uvec), 1.0), 0);

        // TODO In theory, it should reduce the ray length at each step, but it may be unstable with a larger number of samples.
        slice_dir *= rsqrt(acos(float(s_idx) * 10.166 * (MAX_SAMPLES_NUM - 1)));
        
        float theta = PI * r.y;
        float2 angle = 0.0; sincos(theta, angle.x, angle.y);
        float pdf = rsqrt(sqrt(1.0 - angle.x * angle.x)) * rsqrt(2.0);
        slice_dir *= pdf;

		float2 h = -1.0;

		float step_length = step_size;

	    [loop]
	    for (int r_idx = 0; r_idx < MAX_SLICES_NUM; ++r_idx)
	    {
            float ray_seed = step_length * random + float(r_idx);
	        float2 tap = slice_dir.xy * max(ray_seed, float(r_idx) + 1.0) * BUFFER_PIXEL_SIZE;
	        float4 ray_tap = float4(uv + tap, uv - tap);

            if (any(saturate(ray_tap) != ray_tap))
            {
		        break;
   	        }

	        for (int pairs = 0; pairs < 2; ++pairs)
	        {
			    float3 sample_pos = ndc_to_view(pairs == 0 ? ray_tap.xy : ray_tap.zw);
			    float3 sample_center = sample_pos - view_pos;

			    float dist_sqr = dot(sample_center, sample_center);
			    float cos_horizon = rsqrt(dist_sqr) * dot(sample_center, view_dir);

				float cos_phi = 2.0 * (1.0 - cos_horizon);
				cos_phi = (cos_phi * (2.0 - cos_phi));

                float theta = (1.0 - min(0.9705, _Radius));
			    float theta_sqr = theta * theta;
    			float Rd = ((theta_sqr * dist_sqr) / PI) / abs(cos_phi);
    			float Rd2 = saturate(Rd * Rd);
    			float influence = Rd2 / (1.0 - Rd2 + 0.5);
                float max_horizon = cos_horizon - influence;

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

        // clamp horizon to hemisphere
	    h.x = gamma + max(-h.x - gamma, -H_PI);
	    h.y = gamma + min( h.y - gamma,  H_PI);

        // flip horizons on the next direction
	    h = s_idx > 0 ? h.yx : h.xy;

        // [0, 1]
	    h = saturate((h + theta_sign + H_PI) / PI);

        static const uint BITFIELD_SECTORS_NUM = 32u;

        // transform horizon direction into bit field index in [0, ID]
        uint2 v_id = uint2(round(h * BITFIELD_SECTORS_NUM));

        uint bitfield_mask = 0xffffffffu; // https://issues.angleproject.org/issues/353039526
        uint v_min = v_id.x < BITFIELD_SECTORS_NUM ? bitfield_mask << v_id.x : 0u;
        uint v_max = v_id.y != 0u ? bitfield_mask >> (BITFIELD_SECTORS_NUM - v_id.y) : 0u;

        uint current_bitfield = v_min & v_max; 
	    occluded_bitfield |= current_bitfield;

	    if (occluded_bitfield == bitfield_mask) break; // early loop termination when all bitfield sectors are filled

	    float visible_sectors = BITFIELD_SECTORS_NUM - countbits(occluded_bitfield);

	    visibility = visible_sectors / float(BITFIELD_SECTORS_NUM);      
	    visibility *= MAX_SAMPLES_NUM * H_PI * _Intensity;
    }

    visibility = saturate(visibility / (float)MAX_SAMPLES_NUM);

    tex2Dstore(stRawOcclusion, id.xy, float4(1.0 - visibility, 1, 1, 1));
}

void kalman_online(inout float4 value, in sampler2D s, float2 uv)
{
	float2 uv_previous = uv + tex2D(SamplerMotionVectors, uv).rg;
	float4 moments = tex2D(sPackedHistory, uv_previous);
	
	const float learning_rate = _LearningRate;
	
    // A very lax approximation of kalman via moving average..
	float prediction = value.x - moments.x;
	float covariance = lerp(moments.y, prediction, learning_rate);
	float noise_ratio = sqrt(abs(dot(prediction, prediction)));
	float signal_rate = lerp(moments.z, noise_ratio, learning_rate);
	
	float min_gain = saturate(0.0625 * covariance / (signal_rate + 1e-5));
	float process_noise = clamp(signal_rate / learning_rate, min_gain, 1.0);
	
    if (!all(uv_previous > 0 && uv_previous < BUFFER_SCREEN_SIZE))
    {
        signal_rate = 1;
    }
	
	value.x = lerp(moments.x, value.x, process_noise);
	value.y = abs(prediction);
	value.z = signal_rate;
	value.w = 1;
}

float resample(sampler2D s, float2 uv)
{      
    float2 offset;

    // each pixel receives a pure random and deterministic position within the sample
    float dir = weyl_1d(uv * BUFFER_SCREEN_SIZE);
    offset = weyl_2d(cos(dir / TAU)); 
    offset = erfinv(offset * 2.0 - 1.0) * sqrt(2.0);

    float a = tex2Dlod(s, float4(uv + float2( offset.x,  offset.y) * BUFFER_PIXEL_SIZE, 0, 0)).r;
    float b = tex2Dlod(s, float4(uv + float2(-offset.x, -offset.y) * BUFFER_PIXEL_SIZE, 0, 0)).r;
    float c = tex2Dlod(s, float4(uv,                                                    0, 0)).r;
    float d = tex2Dlod(s, float4(uv + float2(-offset.y,  offset.x) * BUFFER_PIXEL_SIZE, 0, 0)).r;
    float e = tex2Dlod(s, float4(uv + float2( offset.y, -offset.x) * BUFFER_PIXEL_SIZE, 0, 0)).r;

    return (a + b + c + d + e) * 0.2;  
}

void guided_filtering(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 output : SV_Target)
{
    float4 v = 1.0;    
    v.x = resample(sRawOcclusion, texcoord);
    kalman_online(v, sRawOcclusion, texcoord);    
    output = v;
}

void pack_history(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 data : SV_Target)
{
    data = tex2D(sReferanceOcclusion, texcoord);
}

float pow2(float x)
{
    return x * x;
}

void main(float4 vpos : SV_Position, float2 uv : TEXCOORD, out float3 output : SV_Target)
{
    float3 color = tex2Dfetch(ReShade::BackBuffer, int2(vpos.xy), 0).rgb;
    
    float k = 0.0625, lambda = 0.5;
    
    float N = tex2D(sReferanceOcclusion, uv + float2(0.0, 1.5) * BUFFER_PIXEL_SIZE);
    float S = tex2D(sReferanceOcclusion, uv - float2(0.0, 1.5) * BUFFER_PIXEL_SIZE);
    float O = tex2D(sReferanceOcclusion, uv);
    float E = tex2D(sReferanceOcclusion, uv + float2(1.5, 0.0) * BUFFER_PIXEL_SIZE);
    float W = tex2D(sReferanceOcclusion, uv - float2(1.5, 0.0) * BUFFER_PIXEL_SIZE);
    
    float cN = exp(-pow2(abs(N - O)/k));
    float cS = exp(-pow2(abs(S - O)/k));
    float cE = exp(-pow2(abs(E - O)/k));
    float cW = exp(-pow2(abs(W - O)/k));
    
    float occlusion = O + lambda * (cN*(N-O) + cS*(S-O) + cE*(E-O) + cW*(W-O));

    float fadefactor = get_fade_factor(uv);
    occlusion = lerp(occlusion, 1.0, fadefactor);

    color *= color;
    color = -log2(max(1e-6, 1.0 - color));
    color *= occlusion;
    color = 1.0 - exp2(-color);
    color = sqrt(color);

    output = _Debug ? occlusion.xxx : color;
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
	    RenderTarget = texReferanceOcclusion;
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