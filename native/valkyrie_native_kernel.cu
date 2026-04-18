#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// ─── Bilinear upsample ────────────────────────────────────────────────────────
__global__ void upsample_bilinear_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w,
    float scale_x, float scale_y,
    float sharpen)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;

    float src_x = (x + 0.5f) / scale_x - 0.5f;
    float src_y = (y + 0.5f) / scale_y - 0.5f;
    int x0 = max(0, min((int)src_x, in_w - 1));
    int y0 = max(0, min((int)src_y, in_h - 1));
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);
    float wx = src_x - x0;
    float wy = src_y - y0;

    for (int c = 0; c < 3; ++c) {
        float v00 = src[(y0 * in_w + x0) * 3 + c];
        float v01 = src[(y0 * in_w + x1) * 3 + c];
        float v10 = src[(y1 * in_w + x0) * 3 + c];
        float v11 = src[(y1 * in_w + x1) * 3 + c];
        float v = (1.f - wy) * ((1.f - wx) * v00 + wx * v01)
                +       wy  * ((1.f - wx) * v10 + wx * v11);
        // Unsharp mask approximation
        float center = src[(y0 * in_w + x0) * 3 + c];
        v += sharpen * (v - center) * 0.15f;
        dst[(y * out_w + x) * 3 + c] = (uint8_t)max(0.f, min(255.f, v));
    }
}

// ─── Temporal blend with motion-adaptive weight ───────────────────────────────
__global__ void temporal_blend_kernel(
    const uint8_t* __restrict__ cur,
    const uint8_t* __restrict__ prev,
    uint8_t* __restrict__ out,
    int n_pixels,
    float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels * 3) return;
    int pixel = idx / 3;
    int c = idx % 3;
    float diff = fabsf((float)cur[pixel * 3 + c] - (float)prev[pixel * 3 + c]);
    // Reduce blending where motion is large to avoid ghosting
    float motion_alpha = alpha * max(0.f, 1.f - diff / 64.f);
    float v = (1.f - motion_alpha) * cur[pixel * 3 + c]
            +       motion_alpha   * prev[pixel * 3 + c];
    out[pixel * 3 + c] = (uint8_t)max(0.f, min(255.f, v));
}

// ─── Bilateral-style denoising kernel ────────────────────────────────────────
__global__ void denoise_bilateral_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int h, int w,
    float sigma_space,
    float sigma_color)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int radius = 3;
    float inv_2ss = 1.f / (2.f * sigma_space * sigma_space);
    float inv_2sc = 1.f / (2.f * sigma_color * sigma_color);

    for (int c = 0; c < 3; ++c) {
        float sum_w = 0.f, sum_v = 0.f;
        float center = src[(y * w + x) * 3 + c];
        for (int dy = -radius; dy <= radius; ++dy) {
            int yy = max(0, min(y + dy, h - 1));
            for (int dx = -radius; dx <= radius; ++dx) {
                int xx = max(0, min(x + dx, w - 1));
                float neighbor = src[(yy * w + xx) * 3 + c];
                float space_d2 = (float)(dx * dx + dy * dy);
                float color_d2 = (center - neighbor) * (center - neighbor);
                float weight = expf(-space_d2 * inv_2ss - color_d2 * inv_2sc);
                sum_w += weight;
                sum_v += weight * neighbor;
            }
        }
        dst[(y * w + x) * 3 + c] = (uint8_t)max(0.f, min(255.f, sum_v / max(sum_w, 1e-6f)));
    }
}

// ─── ACES tonemapping kernel (SDR→HDR expansion) ──────────────────────────────
__global__ void hdr_aces_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int n_pixels,
    float strength,
    float exposure)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels * 3) return;

    float v = src[idx] / 255.f;
    // Linearize
    v = powf(max(v, 0.f), 2.2f);
    v *= exposure;
    // ACES filmic
    float a = 2.51f, b = 0.03f, cc = 2.43f, d = 0.59f, e = 0.14f;
    float tm = (v * (a * v + b)) / (v * (cc * v + d) + e);
    tm = max(0.f, min(1.f, tm));
    // Re-encode gamma
    tm = powf(tm, 1.f / 2.2f);
    // Blend with original
    float result = src[idx] / 255.f * (1.f - strength) + tm * strength;
    dst[idx] = (uint8_t)max(0.f, min(255.f, result * 255.f));
}

// ─── Passthrough (used for testing / no-op path) ─────────────────────────────
extern "C" __global__ void valkyrie_passthrough_kernel(
    const unsigned char* input,
    unsigned char* output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = input[idx];
}
