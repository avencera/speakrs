#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_fp16.h>

// one independent partial per thread avoids atomics and float count loss
__device__ float at(const float* x, uint64_t plane, int h, int w, int y, int z) {
    return y >= 0 && y < h && z >= 0 && z < w ? x[plane + y*w + z] : 0.0f;
}
__device__ float row(float a, float b, float c, float d, int r) {
    if (r == 0) return a-c;
    if (r == 1) return b+c;
    if (r == 2) return c-b;
    return b-d;
}
extern "C" __global__ void range_stats(const float* x, uint64_t len, int h, int w, int transform, uint64_t* out) {
    uint64_t tid = blockIdx.x*blockDim.x+threadIdx.x;
    uint64_t step = gridDim.x*blockDim.x;
    uint64_t count = len;
    int th = (h+1)/2, tw = (w+1)/2;
    if (transform) count = len/(h*w)*th*tw*16;
    float maximum = 0;
    uint64_t small = 0, over = 0, zeros = 0, nonfinite = 0, scaled_small = 0;
    float minimum = INFINITY;
    for (uint64_t i = tid; i < count; i += step) {
        float v;
        if (!transform) v = x[i];
        else {
            int e = i%16;
            uint64_t tile = i/16;
            int tx = tile%tw*2-1, ty = tile/tw%th*2-1;
            uint64_t base = tile/(tw*th)*h*w;
            float r[4];
            for (int c=0; c<4; ++c) r[c] = row(at(x,base,h,w,ty,tx+c), at(x,base,h,w,ty+1,tx+c), at(x,base,h,w,ty+2,tx+c), at(x,base,h,w,ty+3,tx+c), e/4);
            v = row(r[0],r[1],r[2],r[3],e%4);
        }
        float a = fabsf(v);
        maximum = fmaxf(maximum,a);
        if (a > 0) minimum = fminf(minimum,a);
        scaled_small += a > 0 && a < 0x1p-24f;
        zeros += a == 0;
        small += a > 0 && a < 0x1p-14f;
        over += a > 65504.0f;
        nonfinite += !isfinite(v);
    }
    out[tid*8] = __float_as_uint(maximum);
    out[tid*8+1] = small;
    out[tid*8+2] = over;
    out[tid*8+3] = zeros;
    out[tid*8+4] = nonfinite;
    out[tid*8+5] = tid < count ? (count-1-tid)/step+1 : 0;
    out[tid*8+6] = scaled_small;
    out[tid*8+7] = __float_as_uint(minimum);
}

// only conv2 reads the hidden outputs that this diagnostic rounds
extern "C" __global__ void round_stores(float* x, uint64_t len) {
    uint64_t i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < len) x[i] = __half2float(__float2half_rn(x[i]*1024.0f))*(1.0f/1024.0f);
}
