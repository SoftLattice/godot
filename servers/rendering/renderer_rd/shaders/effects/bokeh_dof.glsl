#[compute]

#version 450

#VERSION_DEFINES

#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_control_flow_attributes : require

#define BLOCK_SIZE 8u // Must change tid_to_gpos if != 8u
#define TILE_SIZE (BLOCK_SIZE * BLOCK_SIZE)

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;

#ifdef MODE_GEN_BLUR_SIZE
layout(rgba16f, set = 0, binding = 0) uniform restrict image2D color_image;
layout(set = 1, binding = 0) uniform sampler2D source_depth;
#endif

#if defined(MODE_BOKEH_BOX) || defined(MODE_BOKEH_HEXAGONAL) || defined(MODE_BOKEH_CIRCULAR)
layout(set = 1, binding = 0) uniform sampler2D color_texture;
layout(rgba16f, set = 0, binding = 0) uniform restrict writeonly image2D bokeh_image;
#endif

#ifdef MODE_COMPOSITE_BOKEH
layout(rgba16f, set = 0, binding = 0) uniform restrict image2D color_image;
layout(set = 1, binding = 0) uniform sampler2D source_bokeh;
#endif

#if defined(MODE_BOKEH_CIRCULAR) || defined(MODE_BOKEH_HEXAGONAL)
shared vec4 scratch[(4u * TILE_SIZE)];
#endif

#if defined(MODE_BOKEH_BOX)
shared vec4 scratch[(2u * TILE_SIZE)];
#endif

// based on https://www.shadertoy.com/view/Xd3GDl

#include "bokeh_dof_inc.glsl"

#ifdef MODE_GEN_BLUR_SIZE

float get_depth_at_pos(vec2 uv) {
	float depth = textureLod(source_depth, uv, 0.).x * 2.0 - 1.0;
	if (params.orthogonal) {
		depth = -(depth * (params.z_far - params.z_near) - (params.z_far + params.z_near)) / 2.0;
	} else {
		depth = 2.0 * params.z_near * params.z_far / (params.z_far + params.z_near + depth * (params.z_far - params.z_near));
	}
	return depth;
}

float get_blur_size(float depth) {
	if (params.blur_near_active && depth < params.blur_near_begin) {
		if (params.use_physical_near) {
			// Physically-based.
			float d = abs(params.blur_near_begin - depth);
			return -(d / (params.blur_near_begin - d)) * params.blur_size_near - DEPTH_GAP; // Near blur is negative.
		} else {
			// Non-physically-based.
			return -(1.0 - smoothstep(params.blur_near_end, params.blur_near_begin, depth)) * params.blur_size - DEPTH_GAP; // Near blur is negative.
		}
	}

	if (params.blur_far_active && depth > params.blur_far_begin) {
		if (params.use_physical_far) {
			// Physically-based.
			float d = abs(params.blur_far_begin - depth);
			return (d / (params.blur_far_begin + d)) * params.blur_size_far + DEPTH_GAP;
		} else {
			// Non-physically-based.
			return smoothstep(params.blur_far_begin, params.blur_far_end, depth) * params.blur_size + DEPTH_GAP;
		}
	}

	return 0.0;
}

#endif

//////////////////////////////////////////////////////////////////////////
#if defined(MODE_BOKEH_BOX) || defined(MODE_BOKEH_HEXAGONAL) || defined(MODE_BOKEH_CIRCULAR)

/**
 * Uses circle of confusion to compute contribution of colori to
 * work item's active color_ref pixel
 *
 * @param color_ref_a The reference color depth for this work item's pixel
 * @param colori_a The depth of pixel being measured for its contribution
 * @param r The distance between pixel ref and i
 * @return The bleed of pixel i into the work item's pixel
 */
float compute_bleed(const float color_ref_a, const float colori_a, const float r) {
#if defined(MODE_BOKEH_CIRCULAR)
	// Do the proper CoC calculation for circular
	const float limit = min(abs(colori_a), abs(min(colori_a, color_ref_a)) * 2.0) - r + 0.5 - DEPTH_GAP;
#else
	// Heuristic CoC otherwise
	const float limit = abs(min(color_ref_a, colori_a)) - r + 0.5 - DEPTH_GAP;
#endif
	return clamp(limit, 0., 1.);
}

// Used by all variants
ivec2 g_pos;
vec4 color_ref; // The color of this pixel

vec4 accumulant; // The accumulated numerator
float count; // The accumulated denominator

/**
 * Use butterfly reduction to average pixels in the quality mask
 * subset of work items. This only uses subgroupShuffles, so if
 * 2^quality >= subgroup_size, (only the case in old *mobile* ARM
 * GPUs probably not Forward+) this approximation will suffer
 *
 * @param pos The UV of the pixel fetched by this work item
 * @param subgroup_size The size of the subgroup
 */
vec4 texture_smooth_quality(const vec2 pos, const uint tid) {
	// Load in the color + depth from the texture
	vec4 colori = textureLod(color_texture, pos, 0.);

	// Maximum butterfly swaps 2^N
	for (uint s = 0u; (s < uint(params.blur_steps)); s++) {
		// Exchange the low quality pixels
		vec4 colorj = subgroupShuffleXor(colori, 1u << s);
		// Butterfly sum the color RGB
		colori += colorj;
	}
	return colori / float(1u << uint(params.blur_steps));
}

//////////////////////////////////////////////////////////////////////////
#ifdef MODE_BOKEH_CIRCULAR

// Writes to a morton z-code, but transposes off-diagonal
// blocks to disrupt horizontal bias in pixel masking
ivec2 tid_to_gpos(uint i) {
	// First compute standard morton code
	uint ix = (i & 1u) | ((i >> 1u) & 2u) | ((i >> 2u) & 4u);
	uint iy = ((i >> 1u) & 1u) | ((i >> 2u) & 2u) | ((i >> 3u) & 4u);
	// Determine which levels are on or off-diagonal
	uint swap_mask = (ix ^ iy) >> 1;
	// Transpose bits as necessary
	return ivec2((ix & ~swap_mask) | (iy & swap_mask), (iy & ~swap_mask | (ix & swap_mask)));
}

#endif // MODE_BOKEH_CIRCULAR
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
#if defined(MODE_BOKEH_BOX) || defined(MODE_BOKEH_HEXAGONAL)

// Simple linear layout
ivec2 tid_to_gpos(uint i) {
	return params.second_pass ? ivec2(gl_LocalInvocationID.xy) : ivec2(gl_LocalInvocationID.yx);
}

#endif // defined(MODE_BOKEH_BOX) || defined(MODE_BOKEH_HEXAGONAL)
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
#ifdef MODE_BOKEH_CIRCULAR

vec2 mask_average_gpos(const uint tid) {
	const uint mask = (1u << uint(params.blur_steps)) - 1u;
	vec2 max_gpos = vec2(tid_to_gpos(tid | mask));

	// If no mask exists
	if (!bool(mask)) {
		return max_gpos;
	}

	vec2 min_gpos = vec2(tid_to_gpos(tid & (~mask)));
	return 0.5 * (min_gpos + max_gpos);
}

/**
 * Calls dense convolution for all N^2 pairs in a target 8x8 tile and
 * the active workgroup's 8x8 output tile where, omitting pairs that match on
 * (i&j) & skip_mask != 0, resulting in N^2/(2^bitCount(skip_mask))
 * actual comparisons
 *
 * @param colori Initial color for the reduction
 * @param g_pos_rel Relative position of this thread within the 8x8 tile
 * @param block_offset Relative position of tile w.r.t. workitem output tile
 * @param subgroup_size Size of the subgroup for register shuffles
 * @param tid Thread ID of the work item -- manually set for D3D12 compatibility
 * @param scratch_offset Relative memory position of tile in scratch
 * @param skip_mask Omit all pairs i,j who satisfy (i&j) & skip_mask != 0
 */
void reduce_block(vec4 colori,
		const ivec2 g_pos_rel,
		const vec2 block_offset,
		const uint subgroup_size,
		const uint tid,
		const uint scratch_offset,
		const uint skip_mask) {
	vec4 row_accumulant = vec4(0.);
	float row_count = 0.;
	float row_mf = 0.;

	// Get the relative position of the current pixel
	vec2 g_posi = mask_average_gpos(tid) + block_offset;

	// Repeat the subgroup reduction for each subgroup. Bit twiddle to skip excluded subgroups
	for (uint sg_delta = 0u; sg_delta < TILE_SIZE; sg_delta = ((sg_delta | skip_mask) + subgroup_size) & (~skip_mask)) {
		// If more than 1 subgroup, rotate apparent subgroup for reads
		if (sg_delta > 0u) {
			// The effective workitem of the rotated subgroup
			const uint tid_new = (tid + sg_delta) % TILE_SIZE;
			// Read new color from shared memory
			colori = scratch[tid_new + scratch_offset];
			// Calculate new (relative) global position
			g_posi = mask_average_gpos(tid_new) + block_offset;
		}

		// Accumulate across subgroup via shuffles. Bit twiddle to skip excluded pairs in subgroup
		// for (uint n = 0u; n < subgroup_size; n = ((n | skip_mask) + 1) & (~skip_mask)) {
		for (uint n = 0u; n < subgroup_size; n += (1u << uint(params.blur_steps))) {
			// Swap color + position, using LSB masks to visit all pairs without register pressure
			if (bool(n & skip_mask)) {
				continue;
			}

			// const uint m = n & (-n);
			vec4 colorj = subgroupShuffleXor(colori, n);
			vec2 g_posj = subgroupShuffleXor(g_posi, n);

			// Get distance from current pixel
			const vec2 d_ij = g_posj - g_pos_rel;
			// Only compute sqrt if needed
			const float r = length(d_ij);
			// Create a falloff mask to reduce edge aliasing
			const float f = clamp(params.blur_size - r, 0., 1.);

			// Accumulate result
			const float mi = compute_bleed(color_ref.a, colorj.a, r);
			row_accumulant += colorj * (f * mi);
			row_mf = fma(f, mi, row_mf);
			row_count += f;
		}
	}
	row_accumulant += color_ref * (row_count - row_mf);
	accumulant += row_accumulant;
	count += row_count;
}

#endif // MODE_BOKEH_CIRCULAR
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
#if defined(MODE_BOKEH_BOX) || defined(MODE_BOKEH_HEXAGONAL)

/**
 * Compute the quality average cluster position
 */
float mask_average_pos(const uint tid) {
	const uint mask = (1u << uint(params.blur_steps)) - 1u;
	const float pos_avg = fma(float(mask), 0.5, float((tid & (~mask)) % BLOCK_SIZE));
	return pos_avg;
}

/**
 * Same as reduce_block, but only accumulates across 1x8 rows of the tile.
 *
 * @param colori Initial color for the reduction
 * @param pos_rel Relative linear position of this thread within row
 * @param block_offset Relative linear distance of row w.r.t. workitem row
 * @param subgroup_size Size of the subgroup for register shuffles
 * @param tid Thread ID of the work item -- manually set for D3D12 compatibility
 * @param scratch_offset Relative memory position of tile in scratch
 * @param skip_mask Omit any pairs i,j who satisfy (i ^ skip_mask) == j
 */
void reduce_row(const vec4 colori,
		const int pos_rel,
		const float block_offset,
		const uint subgroup_size,
		const uint tid,
		const uint scratch_offset,
		const uint skip_mask) {
	// Accumulate across subgroup via shuffles. Bit twiddle to skip excluded pairs in subgroup
	vec4 row_accumulant = vec4(0.);
	float row_count = 0.;
	float row_mf = 0.;

	for (uint n = 0u; n < BLOCK_SIZE; n += (1u << uint(params.blur_steps))) {
		if (bool(n & skip_mask)) {
			break;
		}
		// Swap color + position
		vec4 colorj = subgroupShuffleXor(colori, n);
		float pos = mask_average_pos(tid ^ n) + block_offset;

		// Create a falloff mask to reduce edge aliasing
		const float d_ij = abs(pos - pos_rel);
		const float f = clamp(params.blur_size - d_ij, 0., 1.);

		// Accumulate result
		const float mj = compute_bleed(color_ref.a, colorj.a, d_ij);
		row_accumulant += colorj * (f * mj);
		row_mf = fma(f, mj, row_mf);
		row_count += f;
	}
	row_accumulant += color_ref * (row_count - row_mf);
	accumulant += row_accumulant;
	count += row_count;
}

/**
 * Perform a 1D convolution along vector "dir"
 *
 * @param tid The workitem index for this thread
 * @param subgroup_size Size of the subgroup for register shuffles
 * @param dir A 2D direction along which to take the convolution
 * @result The resulting convolution value
 */
vec4 convolve_dir(const uint tid,
		const vec2 pixel_stride,
		const vec2 uv,
		const uint subgroup_size,
		const vec2 dir) {
	const vec2 dir_stride = dir * pixel_stride;

	// The reference color of the current pixel
	color_ref = textureLod(color_texture, uv, 0.);

	// The number of halo blocks required around the edges
	const int halo_blocks = int((uint(ceil(params.blur_size)) + BLOCK_SIZE - 1u) / BLOCK_SIZE);

	// Initialize the accumulator to small value which will resolve to ref color
	count = 0.0001;
	accumulant = color_ref * count;

// Create macro for easier definition
#define PROCESS_BLOCK_PASS(b)                                                                                    \
	{                                                                                                            \
		const float block_offset = float(b * int(BLOCK_SIZE));                                                   \
		vec4 colori = texture_smooth_quality(uv + (block_offset * dir_stride), tid);                             \
		reduce_row(colori, int(tid % BLOCK_SIZE), block_offset, subgroup_size, tid, ((b & 1u) * TILE_SIZE), 0u); \
	}
	PROCESS_BLOCK_PASS(0)
	for (int b = 1; b < min(8, halo_blocks); b++) {
		PROCESS_BLOCK_PASS(b)
		PROCESS_BLOCK_PASS(-b)
	}
#undef PROCESS_BLOCK_PASS

	// Fetch the edges
	{
		subgroupBarrier();
		const float d_block = float(halo_blocks * int(BLOCK_SIZE));
		vec4 colori = texture_smooth_quality(uv + (-d_block * dir_stride), tid);
		scratch[tid ^ ((2u * BLOCK_SIZE) * (tid & (BLOCK_SIZE / 2u)))] = colori;
		colori = texture_smooth_quality(uv + (d_block * dir_stride), tid);
		scratch[(tid | TILE_SIZE) ^ ((2u * BLOCK_SIZE) * (tid & (BLOCK_SIZE / 2u)))] = colori;
		subgroupBarrier();

		float block_offset = float(halo_blocks * int(BLOCK_SIZE)) * float(float((tid & (BLOCK_SIZE / 2u)) / (BLOCK_SIZE / 4u)) - 1.);

		colori = scratch[tid];

		// Reduce the unique halves
		reduce_row(colori, int(tid % BLOCK_SIZE), block_offset, subgroup_size, tid, 0u, (BLOCK_SIZE / 2u));

		// Reduce the others two halves
		colori = scratch[tid | TILE_SIZE];
		reduce_row(colori, int(tid % BLOCK_SIZE), -block_offset, subgroup_size, tid, TILE_SIZE, (BLOCK_SIZE / 2u));
		colori = scratch[(tid ^ (BLOCK_SIZE / 2u)) | TILE_SIZE];
		reduce_row(colori, int(tid % BLOCK_SIZE), block_offset, subgroup_size, tid ^ (BLOCK_SIZE / 2u), TILE_SIZE, (BLOCK_SIZE / 2u));
	}

	return accumulant / count;
}

#endif // defined(MODE_BOKEH_BOX) || defined(MODE_BOKEH_HEXAGONAL)
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
#ifdef MODE_BOKEH_BOX

/**
 * Perform two 1D convolutions for Minkowski addition equivalent to square tophat.
 * Each pass is performed in 2 stages
 *  1) Inner 8x8 tiles accumulated by all items with same (tid / 8) [row / column]
 *  2) End cap halos accumulated in 3 half passes
 *
 * @param subgroup_size Size of the subgroup for register shuffles
 * @return The final color for this work item's pixel
 */
vec4 convolve_color(const uint subgroup_size) {
	// Figure out uv values to read
	const uint tid = gl_LocalInvocationID.x + BLOCK_SIZE * gl_LocalInvocationID.y;

	// The relative position of the pixel to the 8x8 work tile
	const ivec2 g_pos_rel = tid_to_gpos(tid);

	// Offset it by the global position of this block read
	const vec2 pixel_stride = 1. / params.size;

	// Center pixel read
	const vec2 uv = (vec2(g_pos_rel + ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy)) * pixel_stride) + 0.5 * pixel_stride;
	const vec4 color_out = convolve_dir(tid, pixel_stride, uv, subgroup_size, params.second_pass ? vec2(1., 0.) : vec2(0., 1.));

	g_pos = tid_to_gpos(tid) + ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy);
	return color_out;
}

#endif // MODE_BOKEH_BOX
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
#ifdef MODE_BOKEH_HEXAGONAL

vec4 dual_shear_convolution(const uint tid, const uint subgroup_size) {
	// Relative position of this "pixel"
	const vec2 pixel_stride = 1. / params.size;

#define SHEAR_ANGLE 0.571428571428571
	// The sheared direction (1,tan(±pi/6)) - fudged to 2/3.5 (~1% error)
	// this allows pixel-aligned shear. Sign toggles as even rows: (+), odd: (-)
	const vec2 dir = vec2(1., fma(float(gl_LocalInvocationID.y & 1u), -2. * SHEAR_ANGLE, SHEAR_ANGLE));

	// Apply the shear to the pixel position
	float dy = (float(gl_LocalInvocationID.x) - (float(BLOCK_SIZE / 2u) - 0.5)) * dir.y;

	// Compute the unsheared position of the 1/3 pass with a vertical halo of (BLOCK_SIZE/4)
	// Halo chosen so after 3 passes, vertical pixels include:
	// [-2,-1,0,...,BLOCK_SIZE-1,BLOCK_SIZE,BLOCK_SIZE+1]
	// with shear <= 2 for the block, this guarantees to include [0,BLOCK_SIZE-1]
	const vec2 g_pos_global = vec2(float(gl_GlobalInvocationID.x), int((gl_WorkGroupID.y * gl_WorkGroupSize.y) | (gl_LocalInvocationID.y >> 1u)) - int(BLOCK_SIZE / 4u));

	// Apply shear and convert to UV
	vec2 uv = (g_pos_global + vec2(0., dy) + vec2(0.5)) * pixel_stride;

	// Save registers, write first two passes to shared memory
	scratch[tid | (2u * TILE_SIZE)] = convolve_dir(tid, pixel_stride, uv, subgroup_size, dir);
	uv += vec2(0, int(BLOCK_SIZE / 2u)) * pixel_stride;
	scratch[tid | (3u * TILE_SIZE)] = convolve_dir(tid, pixel_stride, uv, subgroup_size, dir);
	uv += vec2(0, int(BLOCK_SIZE / 2u)) * pixel_stride;

	// Keep 3rd pass in register
	vec4 colork = convolve_dir(tid, pixel_stride, uv, subgroup_size, dir);

	barrier();
	// Read first 2 passes back to register
	vec4 colori = scratch[tid | (2u * TILE_SIZE)];
	vec4 colorj = scratch[tid | tid | (3u * TILE_SIZE)];

	// Store colors to shared to perform on-grid interpolation of sheared pixels
	// Rotate / shear where pixels are stored to row-align for interpolation read
	// even rows shift up 33221100, odd shift up 00112233
	{
		barrier();
		const uint y_shift = ((gl_LocalInvocationID.x >> 1u) % (BLOCK_SIZE / 2u)) ^ (bool(gl_LocalInvocationID.y & 1u) ? 0u : (BLOCK_SIZE / 2u) - 1u);

		// The normal location to write
		uvec2 out_index = uvec2(gl_LocalInvocationID.x, gl_LocalInvocationID.y >> 1u);

		// Apply the y-shear -- careful to handle wrap-around of negative uint
		out_index.y = ((out_index.y - y_shift) % (2u * BLOCK_SIZE));

		// Write pass 1 rotated, offset odd rows by 2 TILEs
		scratch[(out_index.x + (BLOCK_SIZE * out_index.y)) | (2u * TILE_SIZE * (gl_LocalInvocationID.y & 1u))] = colori;

		// Recompute to drop y-wrapping
		out_index.y = (((gl_LocalInvocationID.y >> 1u) + (BLOCK_SIZE / 2u)) - y_shift);
		scratch[(out_index.x + (BLOCK_SIZE * out_index.y)) | (2u * TILE_SIZE * (gl_LocalInvocationID.y & 1u))] = colorj;

		// Translate for final write
		out_index.y += BLOCK_SIZE / 2u;
		scratch[(out_index.x + (BLOCK_SIZE * out_index.y)) | (2u * TILE_SIZE * (gl_LocalInvocationID.y & 1u))] = colork;
		barrier();
	}

	// The two convolutions are complete and in scratch, now perform interpolation
	// to resolve on-grid values
	vec4 color_pos;
	// Perform interpolation for (+) angle
	{
		const float y_shift = float(((gl_LocalInvocationID.x >> 1u) % (BLOCK_SIZE / 2u)) ^ (3u));
		// Lower edge of interpolation
		float l = ((float(gl_LocalInvocationID.x) - 3.5) * SHEAR_ANGLE) - float(BLOCK_SIZE / 4u) + y_shift;
		// Upper edge = l+1, so just ignore
		float m = clamp(-l, 0., 1.);

		// Now apply interpolation
		colori = scratch[tid];
		colorj = scratch[tid + BLOCK_SIZE];

		color_pos = mix(colori, colorj, m);
	}

	vec4 color_neg;
	// Perform interpolation for (-) angle
	{
		// Same as for (+) but y_shift is flipped, and SHEAR_ANGLE is (-)
		const float y_shift = float(((gl_LocalInvocationID.x >> 1u) % (BLOCK_SIZE / 2u)));
		float l = (-(float(gl_LocalInvocationID.x) - 3.5) * SHEAR_ANGLE) - float(BLOCK_SIZE / 4u) + y_shift;
		float m = clamp(-l, 0., 1.);

		// Read from other region of scratch
		colori = scratch[tid | (2u * TILE_SIZE)];
		colorj = scratch[(tid + BLOCK_SIZE) | (2u * TILE_SIZE)];

		color_neg = mix(colori, colorj, m);
	}

	// Apply the pseudo-mask trick
	return vec4(min(color_pos.rgb, color_neg.rgb), 0.5 * (color_pos.a + color_neg.a));
}

/**
 * Perform 3 convolutions: 1x vertical followed by combined sheared transforms
 * at +/- 30 degrees. Sheared convolutions are combined using separable hexagon trick.
 *
 * @param subgroup_size Size of the subgroup for register shuffles
 * @return The final color for this work item's pixel
 */
vec4 convolve_color(const uint subgroup_size) {
	const uint tid = gl_LocalInvocationID.x + BLOCK_SIZE * gl_LocalInvocationID.y;

	// Do the vertical convolution second
	if (params.second_pass) {
		// Offset it by the global position of this block read
		vec4 color_out = dual_shear_convolution(tid, subgroup_size);
		g_pos = ivec2(gl_GlobalInvocationID.xy);
		return color_out;
	} else {
		// The relative position of the pixel to the 8x8 work tile
		const ivec2 g_pos_rel = tid_to_gpos(tid);

		// Offset it by the global position of this block read

		const vec2 pixel_stride = 1. / params.size;

		// Center pixel read
		const vec2 uv = (vec2(g_pos_rel + ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy)) * pixel_stride) + 0.5 * pixel_stride;

		vec4 color_out = convolve_dir(tid, pixel_stride, uv, subgroup_size, vec2(0., 1.));

		g_pos = tid_to_gpos(tid) + ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy);
		return color_out;
	}
}

#endif // MODE_BOKEH_HEXAGONAL
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
#ifdef MODE_BOKEH_CIRCULAR

// Macro to identify which quadrant the index b is in
#define SPLIT_COEFF(b) vec2(float((b & 1u) << 1u) - 1.0, float(b & 2u) - 1.0)

/**
 * Perform a dense top-hat convolution with radius varying according to
 * pixel-wise circle of confusion. Convolution is performed in 4 stages
 *  1) Inner 8x8 tiles accumulated by all tid items
 *  2) Halo edge pairs (w/o corners) accumulated in 1 whole pass + 2 half passes
 *  3) Halo (all) corners accumulated in 1 whole pass + 8 quarter passes
 *
 * @param tid Thread ID of the work item -- manually set for D3D12 compatibility
 * @param subgroup_size Size of the subgroup for register shuffles
 * @return The final color for this work item's pixel
 */
vec4 convolve_color(const uint subgroup_size) {
	const uint tid = gl_LocalInvocationID.x + BLOCK_SIZE * gl_LocalInvocationID.y;
	// The position of the thread in the work group
	const ivec2 g_pos_rel = tid_to_gpos(tid);

	// Figure out uv values to read
	const vec2 pixel_stride = 1.0 / vec2(params.size);
	// Center pixel read
	const vec2 uv = (vec2(g_pos_rel + ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy)) * pixel_stride) + 0.5 * pixel_stride;

	// The reference color of the current pixel
	color_ref = textureLod(color_texture, uv, 0.);

	// The number of halo blocks required around the edges
	const int halo_blocks = int((uint(ceil(params.blur_size)) + BLOCK_SIZE - 1u) / BLOCK_SIZE);

	// Initialize the accumulator to small value which will resolve to ref color
	count = 0.0001;
	accumulant = color_ref * count;

	// For all "internal" blocks
	for (int bx = 1 - halo_blocks; bx < halo_blocks; bx++) {
		for (int by = 1 - halo_blocks; by < halo_blocks; by++) {
			// Where is this block relative to (0,0)
			const vec2 block_offset = vec2(bx * int(BLOCK_SIZE), by * int(BLOCK_SIZE));
			vec4 colori = texture_smooth_quality(uv + (block_offset * pixel_stride), tid);

			[[dont_flatten]]
			if (gl_NumSubgroups > 1u) {
				// Place the pixel in scratch if needed -- alternate placement
				// to avoid post-reduce barrier
				scratch[tid | (((bx ^ by) & 1u) * TILE_SIZE)] = colori;
				barrier();
			}

			reduce_block(colori, g_pos_rel, block_offset, subgroup_size, tid, ((bx ^ by) & 1u) * TILE_SIZE, 0u);
		}
	}

// Define a macro to process edge blocks -- fetches horizontal and vertical edges
// Then does reduction in 6 half passes
#define PROCESS_HALO_PASS(boff)                                                                                                                            \
	{ /* Scope to avoid redefinition of colori */                                                                                                          \
		barrier();                                                                                                                                         \
		/* Place vertical halos in first 2 shared mem tiles */                                                                                             \
		vec4 colori = texture_smooth_quality(uv + (vec2((boff).x, -(boff).y) * pixel_stride), tid);                                                        \
		scratch[tid ^ (2u * (tid & (TILE_SIZE / 2u)))] = colori;                                                                                           \
		colori = texture_smooth_quality(uv + ((boff) * pixel_stride), tid);                                                                                \
		scratch[tid ^ ((TILE_SIZE ^ (tid * 2u)) & TILE_SIZE)] = colori;                                                                                    \
		/* Place horizontal halos in last 2 shared mem tiles */                                                                                            \
		colori = texture_smooth_quality(uv + (vec2(-(boff).y, (boff).x) * pixel_stride), tid);                                                             \
		scratch[(tid ^ ((tid & (TILE_SIZE / 4u)) * 4u)) | (2u * TILE_SIZE)] = colori;                                                                      \
		colori = texture_smooth_quality(uv + ((boff).yx * pixel_stride), tid);                                                                             \
		scratch[(tid | TILE_SIZE) ^ ((tid & (TILE_SIZE / 4u)) * 4u) | (2u * TILE_SIZE)] = colori;                                                          \
		barrier();                                                                                                                                         \
		/* Reductions */                                                                                                                                   \
		colori = scratch[tid];                                                                                                                             \
		reduce_block(colori, g_pos_rel, vec2((boff).x, half_coeff.y * (boff).y), subgroup_size, tid, 0u, TILE_SIZE / 2u);                                  \
		colori = scratch[tid | TILE_SIZE];                                                                                                                 \
		reduce_block(colori, g_pos_rel, vec2((boff).x, -half_coeff.y * (boff).y), subgroup_size, tid, TILE_SIZE, TILE_SIZE / 2u);                          \
		colori = scratch[(tid ^ (TILE_SIZE / 2u)) | TILE_SIZE];                                                                                            \
		reduce_block(colori, g_pos_rel, vec2((boff).x, half_coeff.y * (boff).y), subgroup_size, tid ^ (TILE_SIZE / 2u), TILE_SIZE, TILE_SIZE / 2u);        \
		colori = scratch[tid | (2u * TILE_SIZE)];                                                                                                          \
		reduce_block(colori, g_pos_rel, vec2(half_coeff.x * (boff).y, (boff).x), subgroup_size, tid, (2u * TILE_SIZE), TILE_SIZE / 4u);                    \
		colori = scratch[tid | (3u * TILE_SIZE)];                                                                                                          \
		reduce_block(colori, g_pos_rel, vec2(-half_coeff.x * (boff).y, (boff).x), subgroup_size, tid, (3u * TILE_SIZE), TILE_SIZE / 4u);                   \
		colori = scratch[(tid ^ (TILE_SIZE / 4u)) | (3u * TILE_SIZE)];                                                                                     \
		reduce_block(colori, g_pos_rel, vec2(half_coeff.x * (boff).y, (boff).x), subgroup_size, tid ^ (TILE_SIZE / 4u), (3u * TILE_SIZE), TILE_SIZE / 4u); \
	}

	const vec2 half_coeff = SPLIT_COEFF(tid / (TILE_SIZE / 4u));
	const float edge_offset = float(halo_blocks * int(BLOCK_SIZE));

	// Compute the center edges separately
	PROCESS_HALO_PASS(vec2(0.0, edge_offset));

	// Now read all the halo blocks (except corners)
	for (int b = 1; b < halo_blocks; b++) {
		// Check for early exit
		const vec2 d_off = vec2(max(0, int(BLOCK_SIZE) * (b - 1)), int(BLOCK_SIZE) * (halo_blocks - 1));
		if (dot(d_off, d_off) > (params.blur_size * params.blur_size)) {
			return accumulant / count;
		}
		// Process negative side
		PROCESS_HALO_PASS(vec2(float(-b * int(BLOCK_SIZE)), edge_offset));
		// Process positive side
		PROCESS_HALO_PASS(vec2(float(b * int(BLOCK_SIZE)), edge_offset));
	}

#undef PROCESS_HALO_PASS // Remove the macro

	// Early exit for the corners
	if ((1.4142 * float(int(BLOCK_SIZE) * (halo_blocks - 1))) > params.blur_size) {
		return accumulant / count;
	}

	// Finally fetch all the corners
	{
		float boff = float(halo_blocks * int(BLOCK_SIZE));

		barrier();
		// Twiddle the offsets for [outer-most][vertical-edges][horizontal-edges][inner-most]
		// these take 1, 2, 2, 4 (quarter) passes to accumulate with skip_mask (TILE_SIZE/4u) | (TILE_SIZE/2u)
		[[unroll]]
		for (uint L = 0; L < 4; L++) {
			vec4 colori = texture_smooth_quality(uv + SPLIT_COEFF(L) * boff * pixel_stride, tid);
			scratch[tid | (TILE_SIZE * (((tid / (TILE_SIZE / 4u)) & 3u) ^ L))] = colori;
		}

		barrier();

		// Get the outer-most first -- unique to 4 quadrants
		vec4 colori = scratch[tid];
		reduce_block(colori, g_pos_rel, boff * half_coeff, subgroup_size,
				tid, 0u, (TILE_SIZE / 4u) | (TILE_SIZE / 2u));

		// Reduce the vertical vertical quadrants (2 passes only share within TILE_SIZE/2u mask)
		colori = scratch[tid | TILE_SIZE]; // boff = SPLIT_COEFF[2]
		reduce_block(colori, g_pos_rel, boff * SPLIT_COEFF(2) * half_coeff, subgroup_size,
				tid, TILE_SIZE, (TILE_SIZE / 4u) | (TILE_SIZE / 2u));

		colori = scratch[tid ^ (TILE_SIZE / 4u) | TILE_SIZE]; // boff = SPLIT_COEFF[3]
		reduce_block(colori, g_pos_rel, boff * SPLIT_COEFF(3) * half_coeff, subgroup_size,
				tid ^ (TILE_SIZE / 4u), TILE_SIZE, (TILE_SIZE / 4u) | (TILE_SIZE / 2u));

		// Reduce the horizontal quadrants (2 passes only share within TILE_SIZE/4u mask)
		colori = scratch[tid | (2u * TILE_SIZE)]; // boff = SPLIT_COEFF[1]
		reduce_block(colori, g_pos_rel, boff * SPLIT_COEFF(1) * half_coeff, subgroup_size,
				tid, 2u * TILE_SIZE, (TILE_SIZE / 4u) | (TILE_SIZE / 2u));

		colori = scratch[tid ^ (TILE_SIZE / 2u) | (2u * TILE_SIZE)]; // SPLIT_COEFF[3]
		reduce_block(colori, g_pos_rel, boff * SPLIT_COEFF(3) * half_coeff, subgroup_size,
				tid ^ (TILE_SIZE / 2u), 2u * TILE_SIZE, (TILE_SIZE / 4u) | (TILE_SIZE / 2u));

		// Reduce the inner quadrants in 4 passes -- shared among all quadrants
		[[unroll]]
		for (uint L = 0; L < 4; L++) {
			colori = scratch[tid ^ (L * (TILE_SIZE / 4u)) ^ (3u * TILE_SIZE)];
			reduce_block(colori, g_pos_rel, boff * half_coeff * SPLIT_COEFF(L), subgroup_size,
					tid ^ (L * (TILE_SIZE / 4u)), 3u * TILE_SIZE, (TILE_SIZE / 4u) | (TILE_SIZE / 2u));
		}
	}

	g_pos = g_pos_rel + ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy);
	return accumulant / count;
}

#endif // MODE_BOKEH_CIRCULAR
//////////////////////////////////////////////////////////////////////////

#endif // defined(MODE_BOKEH_BOX) || defined(MODE_BOKEH_HEXAGONAL) || defined(MODE_BOKEH_CIRCULAR)
//////////////////////////////////////////////////////////////////////////

void main() {
#if defined(MODE_BOKEH_CIRCULAR) || defined(MODE_BOKEH_BOX) || defined(MODE_BOKEH_HEXAGONAL)
	// Get the size of the subgroups
	const uint subgroup_size = TILE_SIZE / gl_NumSubgroups;

	const vec4 color = convolve_color(subgroup_size);
	imageStore(bokeh_image, g_pos, color);
	return;
#endif

	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThan(pos, params.size))) { //too large, do nothing
		return;
	}

	vec2 pixel_size = 1.0 / vec2(params.size);
	vec2 uv = vec2(pos) / vec2(params.size);

#ifdef MODE_GEN_BLUR_SIZE
	uv += pixel_size * 0.5;
	//precompute size in alpha channel
	float depth = get_depth_at_pos(uv);
	float size = get_blur_size(depth);

	vec4 color = imageLoad(color_image, pos);
	color.a = size;
	imageStore(color_image, pos, color);
#endif

#ifdef MODE_COMPOSITE_BOKEH

	uv += pixel_size * 0.5;
	vec4 color = imageLoad(color_image, pos);
	vec4 bokeh = texture(source_bokeh, uv);

	float mix_amount = max(abs(min(bokeh.a, color.a)), abs(color.a)) - DEPTH_GAP;
	mix_amount = clamp(mix_amount, 0., 1.);

	color.rgb = mix(color.rgb, bokeh.rgb, mix_amount); //blend between hires and lowres

	color.a = 0; //reset alpha
	imageStore(color_image, pos, color);
#endif
}
