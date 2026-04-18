
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cstddef> // for std::ptrdiff_t

namespace py = pybind11;

namespace {

inline std::uint8_t clamp_byte(float value) {
    return static_cast<std::uint8_t>(std::max(0.0f, std::min(255.0f, value)));
}

py::array_t<std::uint8_t> upsample_rgb(py::array_t<std::uint8_t> input, float scale_factor, float sharpen_strength) {
    auto src = input.unchecked<3>();
    const int in_h = src.shape(0);
    const int in_w = src.shape(1);
    const int channels = src.shape(2);
    const int out_h = std::max(1, static_cast<int>(std::round(in_h * scale_factor)));
    const int out_w = std::max(1, static_cast<int>(std::round(in_w * scale_factor)));

    py::array_t<std::uint8_t> output({out_h, out_w, channels});
    auto dst = output.mutable_unchecked<3>();
    for (int y = 0; y < out_h; ++y) {
        float src_y = static_cast<float>(y) / std::max(scale_factor, 1.0f);
        int y0 = std::clamp(static_cast<int>(src_y), 0, in_h - 1);
        int y1 = std::clamp(y0 + 1, 0, in_h - 1);
        float wy = src_y - static_cast<float>(y0);
        for (int x = 0; x < out_w; ++x) {
            float src_x = static_cast<float>(x) / std::max(scale_factor, 1.0f);
            int x0 = std::clamp(static_cast<int>(src_x), 0, in_w - 1);
            int x1 = std::clamp(x0 + 1, 0, in_w - 1);
            float wx = src_x - static_cast<float>(x0);
            for (int c = 0; c < channels; ++c) {
                float v00 = static_cast<float>(src(y0, x0, c));
                float v01 = static_cast<float>(src(y0, x1, c));
                float v10 = static_cast<float>(src(y1, x0, c));
                float v11 = static_cast<float>(src(y1, x1, c));
                float interpolated = (1.0f - wy) * ((1.0f - wx) * v00 + wx * v01) + wy * ((1.0f - wx) * v10 + wx * v11);
                float sharpened = interpolated + sharpen_strength * (interpolated - 127.5f) * 0.05f;
                dst(y, x, c) = clamp_byte(sharpened);
            }
        }
    }
    return output;
}

py::array_t<std::uint8_t> temporal_blend_rgb(py::array_t<std::uint8_t> current, py::array_t<std::uint8_t> previous, float alpha) {
    auto cur = current.unchecked<3>();
    auto prev = previous.unchecked<3>();
    py::array_t<std::uint8_t> output({cur.shape(0), cur.shape(1), cur.shape(2)});
    auto dst = output.mutable_unchecked<3>();
    for (std::ptrdiff_t y = 0; y < cur.shape(0); ++y) {
        for (std::ptrdiff_t x = 0; x < cur.shape(1); ++x) {
            for (std::ptrdiff_t c = 0; c < cur.shape(2); ++c) {
                float value = (1.0f - alpha) * static_cast<float>(cur(y, x, c)) + alpha * static_cast<float>(prev(y, x, c));
                dst(y, x, c) = clamp_byte(value);
            }
        }
    }
    return output;
}

py::array_t<std::uint8_t> restore_rgb(py::array_t<std::uint8_t> input, float denoise_strength, float sharpen_strength) {
    auto src = input.unchecked<3>();
    py::array_t<std::uint8_t> output({src.shape(0), src.shape(1), src.shape(2)});
    auto dst = output.mutable_unchecked<3>();
    const float center_weight = 1.0f + sharpen_strength;
    const float neighbor_weight = std::max(0.05f, denoise_strength * 0.08f);
    for (std::ptrdiff_t y = 0; y < src.shape(0); ++y) {
        for (std::ptrdiff_t x = 0; x < src.shape(1); ++x) {
            for (std::ptrdiff_t c = 0; c < src.shape(2); ++c) {
                float accum = center_weight * static_cast<float>(src(y, x, c));
                float weight_sum = center_weight;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        if (ky == 0 && kx == 0) {
                            continue;
                        }
                        std::ptrdiff_t yy = std::clamp<ptrdiff_t>(y + ky, static_cast<ptrdiff_t>(0), static_cast<ptrdiff_t>(src.shape(0) - 1));
                        std::ptrdiff_t xx = std::clamp<ptrdiff_t>(x + kx, static_cast<ptrdiff_t>(0), static_cast<ptrdiff_t>(src.shape(1) - 1));
                        accum += neighbor_weight * static_cast<float>(src(yy, xx, c));
                        weight_sum += neighbor_weight;
                    }
                }
                dst(y, x, c) = clamp_byte(accum / weight_sum);
            }
        }
    }
    return output;
}

}  // namespace

PYBIND11_MODULE(valkyrie_native, m) {
    m.def("upsample_rgb", &upsample_rgb, "Upsample RGB frame", py::arg("input"), py::arg("scale_factor"), py::arg("sharpen_strength"));
    m.def("temporal_blend_rgb", &temporal_blend_rgb, "Temporal blending for RGB frames", py::arg("current"), py::arg("previous"), py::arg("alpha"));
    m.def("restore_rgb", &restore_rgb, "Restore RGB frame", py::arg("input"), py::arg("denoise_strength"), py::arg("sharpen_strength"));
}
