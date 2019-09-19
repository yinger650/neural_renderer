#include <iostream>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

// for the older gpus atomicAdd with double arguments does not exist
#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

namespace{
template <typename scalar_t>
__global__ void forward_face_index_map_cuda_kernel_1(
        const scalar_t* __restrict__ faces,
        scalar_t* __restrict__ faces_inv,
        int batch_size,
        int num_faces,
        int image_size) {
    /* batch number, face, number, image size, face[v012][RGB] */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * num_faces) {
        return;
    }
    const int is = image_size;
    const scalar_t* face = &faces[i * 9];
    scalar_t* face_inv_g = &faces_inv[i * 9];

    /* return if backside */
    if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]))
        return;

    /* p[num][xy]: x, y is normalized from [-1, 1] to [0, is - 1]. */
    scalar_t p[3][2];
    for (int num = 0; num < 3; num++) {
        for (int dim = 0; dim < 2; dim++) {
            p[num][dim] = 0.5 * (face[3 * num + dim] * is + is - 1);
        }
    }

    /* compute face_inv */
    scalar_t face_inv[9] = {
        p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
        p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
        p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
    scalar_t face_inv_denominator = (
        p[2][0] * (p[0][1] - p[1][1]) +
        p[0][0] * (p[1][1] - p[2][1]) +
        p[1][0] * (p[2][1] - p[0][1]));
    for (int k = 0; k < 9; k++) {
        face_inv[k] /= face_inv_denominator;
    }
    /* set to global memory */
    for (int k = 0; k < 9; k++) {
        face_inv_g[k] = face_inv[k];
    }
}

template <typename scalar_t>
__global__ void forward_face_index_map_cuda_kernel_2(
        const scalar_t* faces,
        scalar_t* faces_inv,
        int32_t* __restrict__ face_index_map,
        scalar_t* __restrict__ weight_map,
        scalar_t* __restrict__ depth_map,
        scalar_t* __restrict__ face_inv_map,
        int batch_size,
        int num_faces,
        int image_size,
        scalar_t near,
        scalar_t far,
        int return_rgb,
        int return_alpha,
        int return_depth) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int is = image_size;
    const int nf = num_faces;
    const int bn = i / (is * is);
    const int pn = i % (is * is);
    const int yi = pn / is;
    const int xi = pn % is;
    const scalar_t yp = (2. * yi + 1 - is) / is;
    const scalar_t xp = (2. * xi + 1 - is) / is;
    
    const scalar_t* face = &faces[bn * nf * 9] - 9;
    scalar_t* face_inv = &faces_inv[bn * nf * 9] - 9;
    scalar_t depth_min = far;
    int face_index_min = -1;
    scalar_t weight_min[3];
    scalar_t face_inv_min[9];
    for (int fn = 0; fn < nf; fn++) {
        /* go to next face */
        face += 9;
        face_inv += 9;
    
        /* return if backside */
        if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]))
            continue;
    
        /* check [py, px] is inside the face */
        if (((yp - face[1]) * (face[3] - face[0]) < (xp - face[0]) * (face[4] - face[1])) ||
            ((yp - face[4]) * (face[6] - face[3]) < (xp - face[3]) * (face[7] - face[4])) ||
            ((yp - face[7]) * (face[0] - face[6]) < (xp - face[6]) * (face[1] - face[7])))
            continue;
    
        /* compute w = face_inv * p */
        scalar_t w[3];
        w[0] = face_inv[3 * 0 + 0] * xi + face_inv[3 * 0 + 1] * yi + face_inv[3 * 0 + 2];
        w[1] = face_inv[3 * 1 + 0] * xi + face_inv[3 * 1 + 1] * yi + face_inv[3 * 1 + 2];
        w[2] = face_inv[3 * 2 + 0] * xi + face_inv[3 * 2 + 1] * yi + face_inv[3 * 2 + 2];
    
        /* sum(w) -> 1, 0 < w < 1 */
        scalar_t w_sum = 0;
        for (int k = 0; k < 3; k++) {
            w[k] = min(max(w[k], 0.), 1.);
            w_sum += w[k];
        }
        for (int k = 0; k < 3; k++) {
            w[k] /= w_sum;
        }
        /* compute 1 / zp = sum(w / z) */
        const scalar_t zp = 1. / (w[0] / face[2] + w[1] / face[5] + w[2] / face[8]);
        if (zp <= near || far <= zp) {
            continue;
        }
    
        /* check z-buffer */
        if (zp < depth_min) {
            depth_min = zp;
            face_index_min = fn;
            for (int k = 0; k < 3; k++) {
                weight_min[k] = w[k];
            }
            if (return_depth) {
                for (int k = 0; k < 9; k++) {
                    face_inv_min[k] = face_inv[k];
                }
            }
        }
    }
    
    /* set to global memory */
    if (0 <= face_index_min) {
        depth_map[i] = depth_min;
        face_index_map[i] = face_index_min;
        for (int k = 0; k < 3; k++) {
            weight_map[3 * i + k] = weight_min[k];
        }
        if (return_depth) {
            for (int k = 0; k < 9; k++) {
                face_inv_map[9 * i + k] = face_inv_min[k];
            }
        }
    }
}

template <typename scalar_t>
__global__ void forward_texture_sampling_cuda_kernel(
		const scalar_t* faces,
		const scalar_t* textures,
		const int32_t* face_index_map,
		const scalar_t* weight_map,
		const scalar_t* depth_map,
		scalar_t* rgb_map,
		int32_t* sampling_index_map,
        scalar_t* sampling_weight_map,
        size_t batch_size,
        int num_faces,
        int image_size,
        int texture_size,
        scalar_t eps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int face_index = face_index_map[i];
    
    if (face_index >= 0) {
        /*
            from global variables:
            batch number, num of faces, image_size, face[v012][RGB], pixel[RGB], weight[v012],
            texture[ts][ts][ts][RGB], sampling indices[8], sampling_weights[8];
        */
        const int bn = i / (image_size * image_size);
        const int nf = num_faces;
        const int ts = texture_size;
        const scalar_t* face = &faces[(bn * nf + face_index) * 9];
        const scalar_t* texture = &textures[(bn * nf + face_index) * ts * ts * ts * 3];
        scalar_t* pixel = &rgb_map[i * 3];
        const scalar_t* weight = &weight_map[i * 3];
        const scalar_t depth = depth_map[i];
        int32_t* sampling_indices = &sampling_index_map[i * 8];
        scalar_t* sampling_weights = &sampling_weight_map[i * 8];
    
        /* get texture index (float) */
        scalar_t texture_index_float[3];
        for (int k = 0; k < 3; k++) { scalar_t tif = weight[k] * (ts - 1) * (depth / (face[3 * k + 2]));
            tif = max(tif, 0.);
            tif = min(tif, ts - 1 - eps);
            texture_index_float[k] = tif;
        }
    
        /* blend */
        scalar_t new_pixel[3] = {0, 0, 0};
        for (int pn = 0; pn < 8; pn++) {
            scalar_t w = 1;                         // weight
            int texture_index_int[3];            // index in source (int)
            for (int k = 0; k < 3; k++) {
                if ((pn >> k) % 2 == 0) {
                    w *= 1 - (texture_index_float[k] - (int)texture_index_float[k]);
                    texture_index_int[k] = (int)texture_index_float[k];
                }
                else {
                    w *= texture_index_float[k] - (int)texture_index_float[k];
                    texture_index_int[k] = (int)texture_index_float[k] + 1;
                }
            }
    
            int isc = texture_index_int[0] * ts * ts + texture_index_int[1] * ts + texture_index_int[2];
            for (int k = 0; k < 3; k++)
                new_pixel[k] += w * texture[isc * 3 + k];
            sampling_indices[pn] = isc;
            sampling_weights[pn] = w;
        }
        for (int k = 0; k < 3; k++)
            pixel[k] = new_pixel[k];
    }
}

template <typename scalar_t>
__global__ void backward_pixel_map_cuda_kernel(
		const scalar_t* faces,
        int32_t*  face_index_map,
        scalar_t*  rgb_map,
        scalar_t*  alpha_map,
        scalar_t*  depth_map,
        scalar_t*  grad_rgb_map,
        scalar_t*  grad_alpha_map,
        scalar_t*  grad_faces,
        size_t batch_size,
        size_t num_faces,
        int32_t mask_index_from,
        int32_t mask_index_to,
        int image_size,
        scalar_t eps,
        int return_rgb,
        int return_alpha) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * num_faces) {
        return;
    }
    // printf("%d/%d/%d, %d[%d]\n", blockIdx.x, blockDim.x, threadIdx.x, int(i/num_faces), i % num_faces) ;

     /**
     * each cuda block for 1 single face
     * bn: n-th batch
     * fn: n-th face
     * is: 2D range (x,y) in [0, is-1]
     */
    const int bn = i / num_faces;
    const int fn = i % num_faces;
    const int is_fn_in_mask = ((mask_index_from <= fn) && (fn < mask_index_to));
    const int is = image_size;
    const scalar_t* face = &faces[i * 9];
    scalar_t grad_face[9] = {};

    /* check backside */
        /**
     * face => {0,1,2; 3,4,5;,6,7,8}
     *  vtx => {(x1,y1,z1); (x2,y2,z2); (x3,y3,z3)}
     */
    if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]))
        return;
    if (!is_fn_in_mask)
        return;
    /* for each edge */
    for (int edge_num = 0; edge_num < 3; edge_num++) {
        /* set points of target edge */
        /**
         * edge_num = 0,1,2
         *   i.e. edge = 0: pi = {0,1,2}, pp = (face{(0,1), (3,4), (6,7)} * is + is - 1) * .5  =>  {(x0,y0), (x1,y1), (x2,y2)}
         *        edge = 1: pi = {1,2,0}, pp = (face{(3,4), (6,7), (0,1)} * is + is - 1) * .5  =>  {(x1,y1), (x2,y2), (x0,y0)}
         *        edge = 2: pi = {2,0,1}, pp = (face{(6,7), (0,1), (3,4)} * is + is - 1) * .5  =>  {(x2,y2), (x0,y0), (x1,y1)}
         *   pp => float
         */
        int pi[3];
        // Min Wang 2019.2.19
        scalar_t pp[3][3];
        for (int num = 0; num < 3; num++)
            pi[num] = (edge_num + num) % 3;
        for (int num = 0; num < 3; num++) {
            for (int dim = 0; dim < 2; dim++) {
                pp[num][dim] = 0.5 * (face[3 * pi[num] + dim] * is + is - 1);
            }
            pp[num][2] = face[3 * pi[num] + 2];
        }

        /* for dy, dx */
        for (int axis = 0; axis < 2; axis++) {
            /* shift axis */
            /**
             * axis = 0,1, switch x,y coord
             * see edge-0 for example
             *   i.e. axis = 0: p = pp{[(0,0),(0,1)], [(1,0),(1,1)], [(2,0),(2,1)]}  =>  {(x0,y0), (x1,y1), (x2,y2)}
             *        axis = 1: p = pp{[(0,1),(0,0)], [(1,1),(1,0)], [(2,1),(2,0)]}  =>  {(y0,x0), (y1,x1), (y2,x2)}
             */
            scalar_t p[3][3];
            for (int num = 0; num < 3; num++) {
                for (int dim = 0; dim < 2; dim++) {
                    p[num][dim] = pp[num][(dim + axis) % 2];
                }
                p[num][2] = pp[num][2];
            }

            
            /**
             * see x-axis for example: 
             *    x0  <  x1 => d = -1, p0 at left of p1
             *    x0  >= x1 => d = +1, p0 at right of p1
             *    y0  <  y1 => d = +1, p0 below p1
             *    y0  >= y1 => d = -1, p0 above p1
             */
             /* set direction */
            int direction;
            if (axis == 0) {
                if (p[0][0] < p[1][0])
                    direction = -1;
                else
                    direction = 1;
            } else {
                if (p[0][0] < p[1][0])
                    direction = 1;
                else
                    direction = -1;
            }
            // get the depth of three vertices
            scalar_t depth_p0 = p[0][2];
            scalar_t depth_p1 = p[1][2];
            scalar_t depth_p2 = p[2][2];
            
            // calc the weights for depth computing
            /* compute face_inv */
            scalar_t p_inv[9] = {
                p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
                p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
                p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
            scalar_t p_inv_denominator = (
                p[2][0] * (p[0][1] - p[1][1]) +
                p[0][0] * (p[1][1] - p[2][1]) +
                p[1][0] * (p[2][1] - p[0][1]));
            for (int k = 0; k < 9; k++) {
                p_inv[k] /= p_inv_denominator;
            }

            /* along edge */
            int d0_from, d0_to;
            d0_from = max(ceil(min(p[0][0], p[1][0])), 0.);
            d0_to = min(max(p[0][0], p[1][0]), is - 1.);
            for (int d0 = d0_from; d0 <= d0_to; d0++) {
                /* get cross point */
                int d1_in, d1_out;
                const scalar_t d1_cross = (p[1][1] - p[0][1]) / (p[1][0] - p[0][0]) * (d0 - p[0][0]) + p[0][1];
                if (0 < direction)
                    d1_in = floor(d1_cross);
                else
                    d1_in = ceil(d1_cross);
                d1_out = d1_in + direction;

                /* continue if cross point is not shown */
                if (d1_in < 0 || is <= d1_in)
                    continue;
                if (d1_out < 0 || is <= d1_out)
                    continue;

                /* get color of in-pixel and out-pixel */
                scalar_t alpha_in;
                scalar_t alpha_out;
                scalar_t *rgb_in;
                scalar_t *rgb_out;
                int map_index_in, map_index_out;
                if (axis == 0) {
                    map_index_in = bn * is * is + d1_in * is + d0;
                    map_index_out = bn * is * is + d1_out * is + d0;
                }
                else {
                    map_index_in = bn * is * is + d0 * is + d1_in;
                    map_index_out = bn * is * is + d0 * is + d1_out;
                }
                // int is_in_in_mask = ((mask_index_from <= face_index_map[map_index_in]) && (face_index_map[map_index_in] < mask_index_to));
                int is_out_in_mask = ((mask_index_from <= face_index_map[map_index_out]) && (face_index_map[map_index_out] < mask_index_to));
                if (return_alpha) {
                    // alpha_in = alpha_map[map_index_in]; //TODO: alway 1
                    // alpha_out = alpha_map[map_index_out]; //TODO: alpha_out*[out_indexmap]_is_in_mask
                    alpha_in = 1.0; //TODO: alway 1
                    alpha_out = alpha_map[map_index_out] * is_out_in_mask; //TODO: alpha_out*[out_indexmap]_is_in_mask
                }
                if (return_rgb) {
                    rgb_in = &rgb_map[map_index_in * 3];
                    rgb_out = &rgb_map[map_index_out * 3];
                }

                /* out */
                // bool is_in_fn = (face_index_map[map_index_in] == fn); 
                bool is_in_fn = 1;
                if (is_in_fn) {
                    int d1_limit;
                    if (0 < direction)
                        d1_limit = is - 1;
                    else
                        d1_limit = 0;
                    int d1_from = max(min(d1_out, d1_limit), 0);
                    int d1_to = min(max(d1_out, d1_limit), is - 1);
                    /*
                    Only grad_alpha_map is part-splited. index_map and alpha_map are full map.
                    */
                    int32_t*  face_index_map_p;
                    scalar_t* alpha_map_p;
                    scalar_t* grad_alpha_map_p;
                    scalar_t* rgb_map_p;
                    scalar_t* grad_rgb_map_p;
                    int map_offset, map_index_from;
                    if (axis == 0) {
                        map_offset = is;
                        map_index_from = bn * is * is + d1_from * is + d0;
                    }
                    else {
                        map_offset = 1;
                        map_index_from = bn * is * is + d0 * is + d1_from;
                    }
                    face_index_map_p = &face_index_map[map_index_from];
                    if (return_alpha) {
                        alpha_map_p = &alpha_map[map_index_from];
                        grad_alpha_map_p = &grad_alpha_map[map_index_from];
                    }
                    if (return_rgb) {
                        rgb_map_p = &rgb_map[map_index_from * 3];
                        grad_rgb_map_p = &grad_rgb_map[map_index_from * 3];
                    }
                    for (int d1 = d1_from; d1 <= d1_to; d1++) {
                        scalar_t diff_grad = 0;
                        int is_p_in_mask = ((mask_index_from <= *face_index_map_p) && (*face_index_map_p < mask_index_to));
                        // TODO: multiply mask sign
                        if (return_alpha) {
                            diff_grad += (*alpha_map_p * is_p_in_mask - alpha_in) * *grad_alpha_map_p;
                            // diff_grad += (*alpha_map_p - alpha_in) * *grad_alpha_map_p;
                        }
                        if (return_rgb) {
                            for (int k = 0; k < 3; k++)
                            diff_grad += (rgb_map_p[k] * is_p_in_mask - rgb_in[k] * is_fn_in_mask) * grad_rgb_map_p[k];
                            // diff_grad += (rgb_map_p[k] - rgb_in[k]) * grad_rgb_map_p[k];
                    }
                    face_index_map_p += map_offset;
                    if (return_alpha) {
                            alpha_map_p += map_offset;
                            grad_alpha_map_p += map_offset;
                        }
                        if (return_rgb) {
                            rgb_map_p += 3 * map_offset;
                            grad_rgb_map_p += 3 * map_offset;
                        }
                        if (diff_grad <= 0)
                            continue;
                        if (p[1][0] != d0) {
                            scalar_t dist = (p[1][0] - p[0][0]) / (p[1][0] - d0) * (d1 - d1_cross) * 2. / is;
                            dist = (0 < dist) ? dist + eps : dist - eps;
                            grad_face[pi[0] * 3 + (1 - axis)] -= diff_grad / dist;
                        }
                        if (p[0][0] != d0) {
                            scalar_t dist = (p[1][0] - p[0][0]) / (d0 - p[0][0]) * (d1 - d1_cross) * 2. / is;
                            dist = (0 < dist) ? dist + eps : dist - eps;
                            grad_face[pi[1] * 3 + (1 - axis)] -= diff_grad / dist;
                        }
                    }
                }

                /* in */
                {
                    int d1_limit;
                    scalar_t d0_cross2;
                    if ((d0 - p[0][0]) * (d0 - p[2][0]) < 0) {
                        d0_cross2 = (p[2][1] - p[0][1]) / (p[2][0] - p[0][0]) * (d0 - p[0][0]) + p[0][1];
                    }
                    else {
                        d0_cross2 = (p[1][1] - p[2][1]) / (p[1][0] - p[2][0]) * (d0 - p[2][0]) + p[2][1];
                    }
                    if (0 < direction)
                        d1_limit = ceil(d0_cross2);
                    else
                        d1_limit = floor(d0_cross2);
                    int d1_from = max(min(d1_in, d1_limit), 0);
                    int d1_to = min(max(d1_in, d1_limit), is - 1);

                    int* face_index_map_p;
                    scalar_t* depth_map_p;
                    scalar_t* alpha_map_p;
                    scalar_t* grad_alpha_map_p;
                    scalar_t* rgb_map_p;
                    scalar_t* grad_rgb_map_p;
                    int map_index_from;
                    int map_offset;
                    if (axis == 0)
                        map_offset = is;
                    else
                        map_offset = 1;
                    if (axis == 0) {
                        map_index_from = bn * is * is + d1_from * is + d0;
                    }
                    else {
                        map_index_from = bn * is * is + d0 * is + d1_from;
                    }
                    face_index_map_p = &face_index_map[map_index_from] - map_offset;
                    depth_map_p = &depth_map[map_index_from] - map_offset;

                    if (return_alpha) {
                        alpha_map_p = &alpha_map[map_index_from] - map_offset;
                        grad_alpha_map_p = &grad_alpha_map[map_index_from] - map_offset;
                    }
                    if (return_rgb) {
                        rgb_map_p = &rgb_map[map_index_from * 3] - 3 * map_offset;
                        grad_rgb_map_p = &grad_rgb_map[map_index_from * 3] - 3 * map_offset;
                    }

                    for (int d1 = d1_from; d1 <= d1_to; d1++) {
                        face_index_map_p += map_offset;
                        depth_map_p += map_offset;

                        if (return_alpha) {
                            alpha_map_p += map_offset;
                            grad_alpha_map_p += map_offset;
                        }
                        if (return_rgb) {
                            rgb_map_p += 3 * map_offset;
                            grad_rgb_map_p += 3 * map_offset;
                        }
                        // Min Wang 2019.2.19
                        // Add the gradient of Z axis
                        // If the face_index is not fn, but GT_alpha_p is 1, means we need "lift up" the face
                        // If the face_index is fn, but GT_alpha_p wanna be 0, means we need "push down" the face
                        // However, we don't know the 2nd depth value, so whenever we want to lift a face, we push down the top face as well
                        /*
                        if (*face_index_map_p != fn)
                            continue;
                        */

                        scalar_t diff_grad = 0;
                        scalar_t diff_grad_updown = 0;

                        int is_p_in_mask = ((mask_index_from <= *face_index_map_p) && (*face_index_map_p < mask_index_to));
                        
                        if (*face_index_map_p == fn) {
                            //If the top face is itself, the pixel could only from 1 to 0
                            // grad_alpha_map_p = 2(alpha_p - GT_alpha__p)
                            // so grad_alpha_map_p >= 0
                        if (return_alpha) {
                                // diff_grad += (*alpha_map_p - alpha_out) * *grad_alpha_map_p;
                                diff_grad += (*alpha_map_p * is_p_in_mask - alpha_out) * *grad_alpha_map_p;
                        }
                        if (return_rgb) {
                            for (int k = 0; k < 3; k++)
                                    // diff_grad += (rgb_map_p[k] - rgb_out[k]) * grad_rgb_map_p[k];
                                    diff_grad += (rgb_map_p[k] * is_p_in_mask - rgb_out[k]) * grad_rgb_map_p[k];
                        }
                        if (diff_grad > 0) {
                            if (p[1][0] != d0) {
                                scalar_t dist = (p[1][0] - p[0][0]) / (p[1][0] - d0) * (d1 - d1_cross) * 2. / is;
                                dist = (0 < dist) ? dist + eps : dist - eps;
                                grad_face[pi[0] * 3 + (1 - axis)] -= diff_grad / dist;
                            }
                            if (p[0][0] != d0) {
                                scalar_t dist = (p[1][0] - p[0][0]) / (d0 - p[0][0]) * (d1 - d1_cross) * 2. / is;
                                dist = (0 < dist) ? dist + eps : dist - eps;
                                grad_face[pi[1] * 3 + (1 - axis)] -= diff_grad / dist;
                            }
                        }
                    } else { 
                        // If the top face is not itself, the pixel could only from 0 to 1
                        // so grad_alpha_map_p <=0
                        // when axis=0, each pixel will be scan twice. So we don't need scan any more.
                        // the x,y of p is d1, d0 when axis=0
                        
                        if (axis == 1) continue;
                        if (return_alpha) {
                            diff_grad_updown += (*alpha_map_p * is_p_in_mask - 1) * *grad_alpha_map_p;
                        }
                        if (diff_grad_updown <=0) continue;
                        // get the top face
                        const scalar_t* face_top = &faces[*face_index_map_p * 9];
                        // calc the depth of p (in face[fn])
                        /* compute w = face_inv * p */
                            scalar_t w[3];
                            w[0] = p_inv[3 * 0 + 0] * d1 + p_inv[3 * 0 + 1] * d0 + p_inv[3 * 0 + 2];
                            w[1] = p_inv[3 * 1 + 0] * d1 + p_inv[3 * 1 + 1] * d0 + p_inv[3 * 1 + 2];
                            w[2] = p_inv[3 * 2 + 0] * d1 + p_inv[3 * 2 + 1] * d0 + p_inv[3 * 2 + 2];
                            scalar_t w_sum = 0;
                            for (int k = 0; k < 3; k++) {
                                w[k] = min(max(w[k], 0.), 1.);
                                w_sum += w[k];
                            }
                            for (int k = 0; k < 3; k++) {
                                w[k] /= w_sum;
                            }
                            scalar_t zp = 1. / (w[0] / depth_p0 + w[1] / depth_p1 + w[2] / depth_p2);
                        if (zp > *depth_map_p) {
                            // current face lift up 
                            scalar_t dist_depth_p = *depth_map_p - zp;
                            for (int lift_point=0; lift_point<3; ++lift_point) {
                                int p1 = (lift_point + 1) % 3;
                                int p2 = (lift_point + 2) % 3;
                                scalar_t x0 = p[lift_point][0];
                                scalar_t y0 = p[lift_point][1];
                                scalar_t z0 = p[lift_point][2];
                                scalar_t x1 = p[p1][0];
                                scalar_t y1 = p[p1][1];
                                scalar_t z1 = p[p1][2];
                                scalar_t x2 = p[p2][0];
                                scalar_t y2 = p[p2][1];
                                scalar_t z2 = p[p2][2];
                                scalar_t xp = d1;
                                scalar_t yp = d0;
                                scalar_t h0 = ((x0-x1)*(y2-y1)-(y0-y1)*(x2-x1)) * ((x0-x1)*(y2-y1)-(y0-y1)*(x2-x1)) +
                                              ((x0-x1)*(z2-z1)-(z0-z1)*(x2-x1)) * ((x0-x1)*(z2-z1)-(z0-z1)*(x2-x1)) +
                                              ((y0-y1)*(z2-z1)-(z0-z1)*(y2-y1)) * ((y0-y1)*(z2-z1)-(z0-z1)*(y2-y1));
                                scalar_t hp = ((xp-x1)*(y2-y1)-(yp-y1)*(x2-x1)) * ((xp-x1)*(y2-y1)-(yp-y1)*(x2-x1)) +
                                              ((xp-x1)*(z2-z1)-(zp-z1)*(x2-x1)) * ((xp-x1)*(z2-z1)-(zp-z1)*(x2-x1)) +
                                              ((yp-y1)*(z2-z1)-(zp-z1)*(y2-y1)) * ((yp-y1)*(z2-z1)-(zp-z1)*(y2-y1));
                                scalar_t dist = dist_depth_p * (exp(sqrt(h0/hp))+1);  // log on h0/hp (1, +inf)
                                // scalar_t dist = -exp(-dist_depth_p * sqrt(h0/hp)+1);  // log on distance_p
                                // scalar_t dist = dist_depth_p * sqrt(h0/hp);  // last version
                                // scalar_t dist = dist_depth_p / ((p[p2][0]-p[p1][0])*(d0-p[p1][1])-(d1-p[p1][0])*(p[p2][1]-p[p1][1])) *
                                //         ((p[p2][0]-p[p1][0])*(p[lift_point][1]-p[p1][1])-(p[lift_point][0]-p[p1][0])*(p[p2][1]-p[p1][1]));
                                dist = (0 < dist) ? dist + eps : dist - eps;
                                if (dist > 0 ) {
                                    printf("<%d, %d> in [%d] z:%.2f top_z:%.2f alpha:%d dist:%.2f lift:%d <%.0f %.0f %.0f> <%.0f %.0f %.0f> <%.0f %.0f %.0f>\n",
                                     d1, d0, fn, zp, *depth_map_p, *alpha_map_p, dist, lift_point, 
                                     p[lift_point][0], p[lift_point][1], p[lift_point][2],
                                     p[p1][0], p[p1][1], p[p1][2],
                                     p[p2][0], p[p2][1], p[p2][2]);
                                }
                                grad_face[pi[lift_point] * 3 + 2] -= diff_grad_updown / dist;
                            }
                            // top face push down
                            scalar_t pt[3][3];
                            for (int num = 0; num < 3; num++) {
                                for (int dim = 0; dim < 2; dim++) {
                                    pt[num][dim] = 0.5 * (face_top[3 * num + dim] * is + is - 1);
                                }
                                pt[num][2] = face_top[3 * num + 2];
                            }
                            for (int down_point=0; down_point<3; ++down_point) {
                                int p1 = (down_point + 1) % 3;
                                int p2 = (down_point + 2) % 3;
                                scalar_t x0 = pt[down_point][0];
                                scalar_t y0 = pt[down_point][1];
                                scalar_t z0 = pt[down_point][2];
                                scalar_t x1 = pt[p1][0];
                                scalar_t y1 = pt[p1][1];
                                scalar_t z1 = pt[p1][2];
                                scalar_t x2 = pt[p2][0];
                                scalar_t y2 = pt[p2][1];
                                scalar_t z2 = pt[p2][2];
                                scalar_t xp = d1;
                                scalar_t yp = d0;
                                scalar_t zp = *depth_map_p;
                                scalar_t h0 = ((x0-x1)*(y2-y1)-(y0-y1)*(x2-x1)) * ((x0-x1)*(y2-y1)-(y0-y1)*(x2-x1)) +
                                              ((x0-x1)*(z2-z1)-(z0-z1)*(x2-x1)) * ((x0-x1)*(z2-z1)-(z0-z1)*(x2-x1)) +
                                              ((y0-y1)*(z2-z1)-(z0-z1)*(y2-y1)) * ((y0-y1)*(z2-z1)-(z0-z1)*(y2-y1));
                                scalar_t hp = ((xp-x1)*(y2-y1)-(yp-y1)*(x2-x1)) * ((xp-x1)*(y2-y1)-(yp-y1)*(x2-x1)) +
                                              ((xp-x1)*(z2-z1)-(zp-z1)*(x2-x1)) * ((xp-x1)*(z2-z1)-(zp-z1)*(x2-x1)) +
                                              ((yp-y1)*(z2-z1)-(zp-z1)*(y2-y1)) * ((yp-y1)*(z2-z1)-(zp-z1)*(y2-y1));
                                scalar_t dist = -dist_depth_p * (exp(sqrt(h0/hp))+1);  // log on h0/hp (1, +inf)
                                // scalar_t dist = exp(-dist_depth_p * sqrt(h0/hp)+1); // log on distance_p
                                // scalar_t dist = - dist_depth_p * sqrt(h0/hp); //last version
                                // scalar_t dist = -dist_depth_p / ((pt[p2][0]-pt[p1][0])*(d0-pt[p1][1])-(d1-pt[p1][0])*(pt[p2][1]-pt[p1][1])) *
                                //         ((pt[p2][0]-pt[p1][0])*(pt[down_point][1]-pt[p1][1])-(pt[down_point][0]-pt[p1][0])*(pt[p2][1]-pt[p1][1]));
                                dist = (0 < dist) ? dist + eps : dist - eps;
                                grad_faces[*face_index_map_p * 9 + down_point * 3 + 2] -= diff_grad_updown / dist;
                            }
                        }
                    }
                }
            }
        }
    }
}

    /* set to global gradient variable */
    for (int k = 0; k < 9; k++)
        grad_faces[i * 9 + k] += grad_face[k];
}

template <typename scalar_t>
__global__ void backward_textures_cuda_kernel(
        const int32_t* face_index_map,
        scalar_t* sampling_weight_map,
        int32_t* sampling_index_map,
        scalar_t* grad_rgb_map,
        scalar_t* grad_textures,
        size_t batch_size,
        size_t num_faces,
        int image_size,
        size_t texture_size) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int face_index = face_index_map[i];
    if (0 <= face_index) {
        int is = image_size;
        int nf = num_faces;
        int ts = texture_size;
        int bn = i / (is * is);    // batch number [0 -> bs]
    
        scalar_t* grad_texture = &grad_textures[(bn * nf + face_index) * ts * ts * ts * 3];
        scalar_t* sampling_weight_map_p = &sampling_weight_map[i * 8];
        int* sampling_index_map_p = &sampling_index_map[i * 8];
        for (int pn = 0; pn < 8; pn++) {
            scalar_t w = *sampling_weight_map_p++;
            int isc = *sampling_index_map_p++;
            scalar_t* grad_texture_p = &grad_texture[isc * 3];
            scalar_t* grad_rgb_map_p = &grad_rgb_map[i * 3];
            for (int k = 0; k < 3; k++)
                atomicAdd(grad_texture_p++, w * *grad_rgb_map_p++);
        }
    }
}

template <typename scalar_t>
__global__ void backward_depth_map_cuda_kernel(
        const scalar_t*  faces,
        const scalar_t*  depth_map,
        const int32_t* face_index_map,
        const scalar_t* face_inv_map,
        const scalar_t* weight_map,
        scalar_t*  grad_depth_map,
        scalar_t*  grad_faces,
        size_t batch_size,
        size_t num_faces,
        int image_size) {
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int fn = face_index_map[i];
    if (0 <= fn) {
        const int nf = num_faces;
        const int is = image_size;
        const int bn = i / (is * is);
        const scalar_t* face = &faces[(bn * nf + fn) * 9];
        const scalar_t depth = depth_map[i];
        const scalar_t depth2 = depth * depth;
        const scalar_t* face_inv = &face_inv_map[i * 9];
        const scalar_t* weight = &weight_map[i * 3];
        const scalar_t grad_depth = grad_depth_map[i];
        scalar_t* grad_face = &grad_faces[(bn * nf + fn) * 9];
    
        /* derivative wrt z */
        for (int k = 0; k < 3; k++) {
            const scalar_t z_k = face[3 * k + 2];
            atomicAdd(&grad_face[3 * k + 2], grad_depth * weight[k] * depth2 / (z_k * z_k));
        }
    
        /* derivative wrt x, y */
        scalar_t tmp[3] = {};
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < 3; l++) {
                tmp[k] += -face_inv[3 * l + k] / face[3 * l + 2];
            }
        }
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < 2; l++) {
            // k: point number, l: dimension
            atomicAdd(&grad_face[3 * k + l], -grad_depth * tmp[l] * weight[k] * depth2 * is / 2);
            }
        }
    }
}
}

std::vector<at::Tensor> forward_face_index_map_cuda(
        at::Tensor faces,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor depth_map,
        at::Tensor face_inv_map,
        at::Tensor faces_inv,
        int image_size,
        float near,
        float far,
        int return_rgb,
        int return_alpha,
        int return_depth) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const int threads = 512;
    const dim3 blocks_1 ((batch_size * num_faces - 1) / threads +1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "forward_face_index_map_cuda_1", ([&] {
      forward_face_index_map_cuda_kernel_1<scalar_t><<<blocks_1, threads>>>(
          faces.data<scalar_t>(),
          faces_inv.data<scalar_t>(),
          batch_size,
          num_faces,
          image_size);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_face_index_map_1: %s\n", cudaGetErrorString(err));

    const dim3 blocks_2 ((batch_size * image_size * image_size - 1) / threads +1);
    AT_DISPATCH_FLOATING_TYPES(faces.type(), "forward_face_index_map_cuda_2", ([&] {
      forward_face_index_map_cuda_kernel_2<scalar_t><<<blocks_2, threads>>>(
          faces.data<scalar_t>(),
          faces_inv.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          weight_map.data<scalar_t>(),
          depth_map.data<scalar_t>(),
          face_inv_map.data<scalar_t>(),
          (int) batch_size,
          (int) num_faces,
          (int) image_size,
          (scalar_t) near,
          (scalar_t) far,
          return_rgb,
          return_alpha,
          return_depth);
      }));

    err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_face_index_map_2: %s\n", cudaGetErrorString(err));
    return {face_index_map, weight_map, depth_map, face_inv_map};
}

std::vector<at::Tensor> forward_texture_sampling_cuda( at::Tensor faces,
        at::Tensor textures,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor depth_map,
        at::Tensor rgb_map,
        at::Tensor sampling_index_map,
        at::Tensor sampling_weight_map,
        int image_size,
        float eps) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const auto texture_size = textures.size(2);
    const int threads = 512;
    const dim3 blocks ((batch_size * image_size * image_size - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "forward_texture_sampling_cuda", ([&] {
      forward_texture_sampling_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          textures.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          weight_map.data<scalar_t>(),
          depth_map.data<scalar_t>(),
          rgb_map.data<scalar_t>(),
		  sampling_index_map.data<int32_t>(),
		  sampling_weight_map.data<scalar_t>(),
          batch_size,
		  num_faces,
          image_size,
          texture_size,
          eps);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_texture_sampling: %s\n", cudaGetErrorString(err));

    return {rgb_map, sampling_index_map, sampling_weight_map};
}

at::Tensor backward_pixel_map_cuda(
        at::Tensor faces,
        at::Tensor face_index_map,
        at::Tensor rgb_map,
        at::Tensor alpha_map,
        at::Tensor depth_map,
        at::Tensor grad_rgb_map,
        at::Tensor grad_alpha_map,
        at::Tensor grad_faces,
        int32_t mask_index_from,
        int32_t mask_index_to,
        int image_size,
        float eps,
        int return_rgb,
        int return_alpha) {
    
    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const int threads = 256;
    const dim3 blocks ((batch_size * num_faces - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "backward_pixel_map_cuda", ([&] {
      backward_pixel_map_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          rgb_map.data<scalar_t>(),
          alpha_map.data<scalar_t>(),
          depth_map.data<scalar_t>(),
          grad_rgb_map.data<scalar_t>(),
          grad_alpha_map.data<scalar_t>(),
          grad_faces.data<scalar_t>(),
          batch_size,
          num_faces,
          mask_index_from,
          mask_index_to,
          image_size,
          (scalar_t) eps,
          return_rgb,
          return_alpha);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in backward_pixel_map: %s\n", cudaGetErrorString(err));

    return grad_faces;
}

at::Tensor backward_textures_cuda(
        at::Tensor face_index_map,
        at::Tensor sampling_weight_map,
        at::Tensor sampling_index_map,
        at::Tensor grad_rgb_map,
        at::Tensor grad_textures,
        int num_faces) {

    const auto batch_size = face_index_map.size(0);
    const auto image_size = face_index_map.size(1);
    const auto texture_size = grad_textures.size(2);
    const int threads = 256;
    const dim3 blocks ((batch_size * image_size * image_size - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(sampling_weight_map.type(), "backward_textures_cuda", ([&] {
      backward_textures_cuda_kernel<scalar_t><<<blocks, threads>>>(
          face_index_map.data<int32_t>(),
          sampling_weight_map.data<scalar_t>(),
          sampling_index_map.data<int32_t>(),
          grad_rgb_map.data<scalar_t>(),
          grad_textures.data<scalar_t>(),
          batch_size,
          num_faces,
          image_size,
          texture_size);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in backward_textures: %s\n", cudaGetErrorString(err));

    return grad_textures;
}
at::Tensor backward_depth_map_cuda(
        at::Tensor faces,
        at::Tensor depth_map,
        at::Tensor face_index_map,
        at::Tensor face_inv_map,
        at::Tensor weight_map,
        at::Tensor grad_depth_map,
        at::Tensor grad_faces,
        int image_size) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const int threads = 256;
    const dim3 blocks ((batch_size * image_size * image_size - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "backward_depth_map_cuda", ([&] {
      backward_depth_map_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          depth_map.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          face_inv_map.data<scalar_t>(),
          weight_map.data<scalar_t>(),
          grad_depth_map.data<scalar_t>(),
          grad_faces.data<scalar_t>(),
          batch_size,
          num_faces,
          image_size);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in backward_depth_map: %s\n", cudaGetErrorString(err));

    return grad_faces;
}
