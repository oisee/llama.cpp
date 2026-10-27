#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "repack.h"
#include "traits.h"
#include "ggml-impl.h"
#include "amx/amx.h"
#include "ggml-quants.h"
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>

#include <cctype>
#include <string>
#include <vector>

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#endif

#if defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif

// Portable tier-only integer dots with the CUDA activation representation.
static float ggml_cpu_tier_reciprocal(float x, const float * table) {
    uint32_t bits; std::memcpy(&bits, &x, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const int exponent = (bits >> 23) & 255;
    uint32_t result;
    if (exponent == 0) { result = sign | 0x7f800000u; }
    else if (exponent == 255) { result = (bits & 0x7fffffu) ? bits : sign; }
    else {
        float normalized = table[bits & 0x7fffffu];
        std::memcpy(&result, &normalized, sizeof(result));
        const int scaled_exponent = int(result >> 23) + 127 - exponent;
        if (scaled_exponent <= 0) { result = sign; }
        else if (scaled_exponent >= 255) { result = sign | 0x7f800000u; }
        else { result = sign | (uint32_t(scaled_exponent) << 23) | (result & 0x7fffffu); }
    }
    float value; std::memcpy(&value, &result, sizeof(value)); return value;
}

static float ggml_cpu_tier_ftz(float x) {
    return std::abs(x) < std::numeric_limits<float>::min() ? std::copysign(0.0f, x) : x;
}

static void ggml_cpu_tier_swiglu_scalar(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto & params = *static_cast<const ggml_cpu_tier_swiglu_params *>(userdata);
    const auto * gate = dst->src[0]; const auto * up = dst->src[1];
    GGML_ASSERT(params.reciprocal && gate->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(gate, up) && ggml_are_same_shape(gate, dst) && ggml_is_contiguous(dst));
    GGML_ASSERT(gate->ne[3] == 1);
    auto * output = static_cast<float *>(dst->data);
    for (int64_t i = ith; i < ggml_nelements(dst); i += nth) {
        const int64_t c = i % gate->ne[0], q = i / gate->ne[0];
        const int64_t y = q % gate->ne[1], z = q / gate->ne[1];
        float g, u;
        std::memcpy(&g, static_cast<const char *>(gate->data) + c * gate->nb[0] + y * gate->nb[1] + z * gate->nb[2], sizeof(g));
        std::memcpy(&u, static_cast<const char *>(up->data) + c * up->nb[0] + y * up->nb[1] + z * up->nb[2], sizeof(u));
        g = ggml_cpu_tier_ftz(g); u = ggml_cpu_tier_ftz(u);
        if (params.clamp > 1e-6f && params.glm) { g = std::min(g, params.clamp); }
        const float argument = ggml_cpu_tier_ftz(-g * 1.4426950408889634f);
        float exponential;
        if (params.exponential && std::isfinite(argument) && std::abs(argument) < 128.0f) {
            const int integral = int(std::abs(argument));
            const float fraction = std::abs(argument) - integral;
            const float * thresholds = params.exponential + (argument < 0 ? (size_t(1) << 23) : 0);
            // Invert the sampled output thresholds, then restore the integer exponent.
            const uint32_t index = std::upper_bound(thresholds, thresholds + (size_t(1) << 23), fraction) - thresholds - 1;
            uint32_t bits = argument < 0 ? 0x3f800000u - index : 0x3f800000u + index;
            const int exponent = int(bits >> 23) + (argument < 0 ? -integral : integral);
            if (exponent <= 0) { bits = 0; }
            else if (exponent >= 255) { bits = 0x7f800000u; }
            else { bits = (uint32_t(exponent) << 23) | (bits & 0x7fffffu); }
            std::memcpy(&exponential, &bits, sizeof(exponential));
        } else { exponential = std::exp2(argument); }
        float silu = ggml_cpu_tier_ftz(g * ggml_cpu_tier_reciprocal(1.0f + exponential, params.reciprocal));
        if (params.clamp > 1e-6f) {
            if (!params.glm) { silu = std::min(silu, params.clamp); }
            u = std::clamp(u, -params.clamp, params.clamp);
        }
        output[i] = ggml_cpu_tier_ftz(silu * u);
    }
}

#if defined(__AVX2__) && defined(__FMA__)
static __m256 ggml_cpu_tier_ftz8(__m256 x) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256 tiny = _mm256_cmp_ps(_mm256_andnot_ps(sign, x), _mm256_set1_ps(std::numeric_limits<float>::min()), _CMP_LT_OQ);
    return _mm256_blendv_ps(x, _mm256_and_ps(sign, x), tiny);
}

static __m256 ggml_cpu_tier_reciprocal8(__m256 x, const float * table) {
    const __m256i bits = _mm256_castps_si256(x);
    const __m256i mantissa = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffff));
    const __m256i exponent = _mm256_and_si256(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(255));
    const __m256i normalized = _mm256_castps_si256(_mm256_i32gather_ps(table, mantissa, 4));
    const __m256i scaled = _mm256_sub_epi32(_mm256_add_epi32(_mm256_srli_epi32(normalized, 23), _mm256_set1_epi32(127)), exponent);
    __m256i result = _mm256_or_si256(_mm256_slli_epi32(scaled, 23), _mm256_and_si256(normalized, _mm256_set1_epi32(0x7fffff)));
    result = _mm256_andnot_si256(_mm256_cmpgt_epi32(_mm256_set1_epi32(1), scaled), result);
    result = _mm256_blendv_epi8(result, _mm256_set1_epi32(0x7f800000), _mm256_cmpgt_epi32(scaled, _mm256_set1_epi32(254)));
    result = _mm256_or_si256(result, _mm256_and_si256(bits, _mm256_set1_epi32(int32_t(0x80000000u))));
    __m256 out = _mm256_castsi256_ps(result);
    const __m256i special = _mm256_or_si256(_mm256_cmpeq_epi32(exponent, _mm256_setzero_si256()), _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(255)));
    if (_mm256_movemask_epi8(special)) {
        float inputs[8], outputs[8]; _mm256_storeu_ps(inputs, x); _mm256_storeu_ps(outputs, out);
        const int mask = _mm256_movemask_ps(_mm256_castsi256_ps(special));
        for (int i = 0; i < 8; ++i) { if (mask & (1 << i)) { outputs[i] = ggml_cpu_tier_reciprocal(inputs[i], table); } }
        out = _mm256_loadu_ps(outputs);
    }
    return out;
}
#endif

void ggml_cpu_tier_swiglu(ggml_tensor * dst, int ith, int nth, void * userdata) {
#if defined(__AVX2__) && defined(__FMA__)
    const auto & params = *static_cast<const ggml_cpu_tier_swiglu_params *>(userdata);
    const auto * gate = dst->src[0]; const auto * up = dst->src[1];
    if (std::getenv("CNM_TIER_CUDA_SCALAR") || !params.exponential || !ggml_is_contiguous(gate) || !ggml_is_contiguous(up) || ggml_nelements(dst) % 8) {
        ggml_cpu_tier_swiglu_scalar(dst, ith, nth, userdata); return;
    }
    GGML_ASSERT(params.reciprocal && gate->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(gate, up) && ggml_are_same_shape(gate, dst) && ggml_is_contiguous(dst) && gate->ne[3] == 1);
    const auto * gates = static_cast<const float *>(gate->data);
    const auto * ups = static_cast<const float *>(up->data);
    auto * output = static_cast<float *>(dst->data);
    const int64_t blocks = ggml_nelements(dst) / 8;
    for (int64_t block = blocks * ith / nth; block < blocks * (ith + 1) / nth; ++block) {
        const int64_t i = block * 8;
        __m256 g = ggml_cpu_tier_ftz8(_mm256_loadu_ps(gates + i));
        __m256 u = ggml_cpu_tier_ftz8(_mm256_loadu_ps(ups + i));
        const __m256 limit = _mm256_set1_ps(params.clamp);
        if (params.clamp > 1e-6f && params.glm) { g = _mm256_min_ps(g, limit); }
        const __m256 argument = ggml_cpu_tier_ftz8(_mm256_mul_ps(g, _mm256_set1_ps(-1.4426950408889634f)));
        const __m256 integral = _mm256_round_ps(argument, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        const __m256 fraction = _mm256_sub_ps(argument, integral);
        // This polynomial only seeds an index. Stored CUDA thresholds decide every output bit.
        __m256 seed = _mm256_set1_ps(0.00000010178086009239699f);
        for (float coefficient : {0.0000013215486790144305f, 0.00001525273380405984f, 0.000154035303933816f,
                0.001333355814642844f, 0.009618129107628477f, 0.05550410866482158f,
                0.2402265069591007f, 0.6931471805599453f, 1.0f}) {
            seed = _mm256_add_ps(_mm256_mul_ps(seed, fraction), _mm256_set1_ps(coefficient));
        }
        float arguments[8], seeds[8], exponentials[8];
        _mm256_storeu_ps(arguments, argument); _mm256_storeu_ps(seeds, seed);
        for (int j = 0; j < 8; ++j) {
            const float arg = arguments[j];
            if (!std::isfinite(arg) || std::abs(arg) >= 128.0f) { exponentials[j] = std::exp2(arg); continue; }
            const int whole = int(std::abs(arg));
            const float f = std::abs(arg) - whole;
            const float * thresholds = params.exponential + (arg < 0 ? (size_t(1) << 23) : 0);
            uint32_t bits; std::memcpy(&bits, seeds + j, sizeof(bits));
            int index = std::clamp(arg < 0 ? int(0x3f800000u) - int(bits) : int(bits) - int(0x3f800000u), 0, (1 << 23) - 1);
            while (index > 0 && thresholds[index] > f) { --index; }
            while (index + 1 < (1 << 23) && thresholds[index + 1] <= f) { ++index; }
            bits = arg < 0 ? 0x3f800000u - index : 0x3f800000u + index;
            const int exponent = int(bits >> 23) + (arg < 0 ? -whole : whole);
            if (exponent <= 0) { bits = 0; }
            else if (exponent >= 255) { bits = 0x7f800000u; }
            else { bits = (uint32_t(exponent) << 23) | (bits & 0x7fffffu); }
            std::memcpy(exponentials + j, &bits, sizeof(bits));
        }
        const __m256 denominator = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_loadu_ps(exponentials));
        __m256 silu = ggml_cpu_tier_ftz8(_mm256_mul_ps(g, ggml_cpu_tier_reciprocal8(denominator, params.reciprocal)));
        if (params.clamp > 1e-6f) {
            if (!params.glm) { silu = _mm256_min_ps(silu, limit); }
            u = _mm256_max_ps(_mm256_min_ps(u, limit), _mm256_set1_ps(-params.clamp));
        }
        _mm256_storeu_ps(output + i, ggml_cpu_tier_ftz8(_mm256_mul_ps(silu, u)));
    }
#else
    ggml_cpu_tier_swiglu_scalar(dst, ith, nth, userdata);
#endif
}

static void ggml_cpu_tier_quant_matmul_scalar(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto & config = *static_cast<const ggml_cpu_tier_quant_policy *>(userdata);
    const int policy = config.kind;
    GGML_ASSERT(policy == 1 || policy == 2);
    const int lanes = config.mmvq_warps == 4 ? 128 : 32;
    const auto * weight = dst->src[0];
    GGML_ASSERT(weight->type == GGML_TYPE_IQ3_S || weight->type == GGML_TYPE_IQ3_XXS ||
                weight->type == GGML_TYPE_IQ4_XS || weight->type == GGML_TYPE_IQ4_NL ||
                weight->type == GGML_TYPE_IQ2_S || weight->type == GGML_TYPE_IQ2_XS || weight->type == GGML_TYPE_Q2_0);
    const auto * input = dst->src[1];
    const auto * ids = dst->src[2];
    const auto * traits = ggml_get_type_traits(weight->type);
    const int64_t width = weight->ne[0], rows = weight->ne[1], block = traits->blck_size;
    GGML_ASSERT(width % 32 == 0 && width % block == 0 && block <= 256);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && input->ne[0] == width && input->ne[1] == 1);
    GGML_ASSERT(input->ne[3] == 1 && ids->ne[2] == 1 && ids->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[0] == 1 && ids->ne[1] == input->ne[2]);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->ne[0] == rows && dst->ne[1] == 1 && dst->ne[2] == input->ne[2]);
    std::vector<int8_t> codes(width);
    std::vector<float> scales(width / 32);
    std::array<float, 256> decoded;
    for (int64_t pair = 0; pair < input->ne[2]; ++pair) {
        const auto * x = reinterpret_cast<const float *>(static_cast<const char *>(input->data) + pair * input->nb[2]);
        int32_t expert;
        std::memcpy(&expert, static_cast<const char *>(ids->data) + pair * ids->nb[1], sizeof(expert));
        GGML_ASSERT(expert >= 0 && expert < weight->ne[2]);
        for (int64_t c = 0; c < width; c += 32) {
            float amax = 0;
            for (int i = 0; i < 32; ++i) { amax = std::max(amax, std::abs(x[c + i])); }
            const float d = config.reciprocal ? amax * (1.0f / 127.0f) : amax / 127.0f;
            const float inverse = amax ? (config.reciprocal ? 127.0f * ggml_cpu_tier_reciprocal(amax, config.reciprocal) : 127.0f / amax) : 0;
            scales[c / 32] = policy == 1 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(d)) :
                (amax ? (config.reciprocal ? ggml_cpu_tier_reciprocal(inverse, config.reciprocal) : 1.0f / inverse) : 0);
            for (int i = 0; i < 32; ++i) {
                codes[c + i] = amax ? int8_t(std::round(policy == 1 ?
                    (config.reciprocal ? x[c + i] * ggml_cpu_tier_reciprocal(d, config.reciprocal) : x[c + i] / d) : x[c + i] * inverse)) : 0;
            }
        }
        auto * out = reinterpret_cast<float *>(static_cast<char *>(dst->data) + pair * dst->nb[2]);
        for (int64_t row = ith; row < rows; row += nth) {
            const auto * raw = static_cast<const char *>(weight->data) + expert * weight->nb[2] + row * weight->nb[1];
            float sum = 0;
            std::array<float, 128> partial{};
            for (int64_t c = 0; c < width; c += block) {
                const auto * packed = raw + (c / block) * traits->type_size;
                ggml_fp16_t half;
                std::memcpy(&half, packed, sizeof(half));
                float d = ggml_fp16_to_fp32(half);
                if (weight->type == GGML_TYPE_IQ3_XXS) { d *= 0.25f; }
                traits->to_float(packed, decoded.data(), block);
                if (d == 0) { continue; }
                for (int64_t group = 0; group < block; group += 32) {
                    const int g = group / 32;
                    const float activation_scale = scales[(c + group) / 32];
                    if (weight->type == GGML_TYPE_IQ2_S || weight->type == GGML_TYPE_IQ2_XS) {
                        const auto * scales2 = weight->type == GGML_TYPE_IQ2_S ? reinterpret_cast<const block_iq2_s *>(packed)->scales : reinterpret_cast<const block_iq2_xs *>(packed)->scales;
                        const int ls0 = scales2[g] & 15, ls1 = scales2[g] >> 4;
                        const float s0 = (ls0 * d + d / 2) / 4, s1 = (ls1 * d + d / 2) / 4;
                        int dots[2] = {};
                        for (int i = 0; i < 32; ++i) {
                            const float scale = i < 16 ? s0 : s1;
                            dots[i / 16] += int(std::round(decoded[group + i] / scale)) * codes[c + group + i];
                        }
                        if (policy == 2) {
                            sum = std::fma(std::fma(float(dots[0]), s0, float(dots[1]) * s1), activation_scale, sum);
                        } else {
                            const int dot = (dots[0] * ls0 + dots[1] * ls1 + (dots[0] + dots[1]) / 2) / 4;
                            partial[((c + group) / 32) % lanes] = std::fma(d * activation_scale, float(dot), partial[((c + group) / 32) % lanes]);
                        }
                        continue;
                    }
                    float subscale = d;
                    int multiplier = 1;
                    if (weight->type == GGML_TYPE_IQ3_S) {
                        const auto * b = reinterpret_cast<const block_iq3_s *>(packed);
                        multiplier = 1 + 2 * ((b->scales[g / 2] >> (4 * (g % 2))) & 15);
                    } else if (weight->type == GGML_TYPE_IQ4_XS) {
                        const auto * b = reinterpret_cast<const block_iq4_xs *>(packed);
                        multiplier = ((b->scales_l[g / 2] >> (4 * (g % 2))) & 15) | (((b->scales_h >> (2 * g)) & 3) << 4);
                        multiplier -= 32;
                    } else if (weight->type == GGML_TYPE_IQ3_XXS) {
                        const auto * b = reinterpret_cast<const block_iq3_xxs *>(packed);
                        uint32_t scale_signs;
                        std::memcpy(&scale_signs, b->qs + QK_K / 4 + 4 * g, sizeof(scale_signs));
                        multiplier = 1 + 2 * (scale_signs >> 28);
                    }
                    subscale *= multiplier;
                    if (subscale == 0) { continue; }
                    int dots[2] = {};
                    for (int i = 0; i < 32; ++i) {
                        dots[(i % 16) / 8] += int(std::round(decoded[group + i] / subscale)) * codes[c + group + i];
                    }
                    if (policy == 2) {
                        sum = std::fma(float(dots[0] + dots[1]) * subscale, activation_scale, sum);
                    } else if (weight->type == GGML_TYPE_IQ3_XXS) {
                        const int dot = dots[0] + dots[1], ls = (multiplier - 1) / 2;
                        const int scaled = (ls * dot + dot / 2) / 2;
                        partial[((c + group) / 32) % lanes] = std::fma((4 * d) * activation_scale, float(scaled), partial[((c + group) / 32) % lanes]);
                    } else if (weight->type == GGML_TYPE_IQ4_NL) {
                        for (int i = 0; i < 2; ++i) { partial[((c + group) / 16 + i) % lanes] = std::fma(d * activation_scale, float(dots[i]), partial[((c + group) / 16 + i) % lanes]); }
                    } else {
                        partial[((c + group) / 32) % lanes] = std::fma(d * activation_scale, float((dots[0] + dots[1]) * multiplier), partial[((c + group) / 32) % lanes]);
                    }
                }
            }
            if (policy == 1) {
                for (int warp = 1; warp < lanes / 32; ++warp) {
                    for (int i = 0; i < 32; ++i) { partial[i] += partial[warp * 32 + i]; }
                }
                for (int offset = 16; offset; offset /= 2) {
                    auto previous = partial;
                    for (int i = 0; i < 32; ++i) { partial[i] = previous[i] + previous[i ^ offset]; }
                }
                sum = partial[0];
            }
            out[row] = sum;
        }
    }
}

#if defined(__AVX2__) && defined(__FMA__)
template<ggml_type type>
static __m256i ggml_cpu_tier_decode32(const void * packed, int g) {
    if (type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ4_NL) {
        const uint8_t * qs = type == GGML_TYPE_IQ4_XS ? static_cast<const block_iq4_xs *>(packed)->qs + 16 * g : static_cast<const block_iq4_nl *>(packed)->qs;
        const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qs));
        const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_iq4nl));
        const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, _mm_set1_epi8(15)));
        const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), _mm_set1_epi8(15)));
        return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    } else if (type == GGML_TYPE_Q2_0) {
        const auto * b = static_cast<const block_q2_0 *>(packed);
        const __m256i q = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(b->qs + 8 * g)));
        const __m256i lo = _mm256_or_si256(_mm256_and_si256(q, _mm256_set1_epi32(3)), _mm256_and_si256(_mm256_slli_epi32(q, 6), _mm256_set1_epi32(3 << 8)));
        const __m256i hi = _mm256_or_si256(_mm256_and_si256(_mm256_slli_epi32(q, 12), _mm256_set1_epi32(3 << 16)), _mm256_and_si256(_mm256_slli_epi32(q, 18), _mm256_set1_epi32(3 << 24)));
        return _mm256_sub_epi8(_mm256_or_si256(lo, hi), _mm256_set1_epi8(1));
    } else {
        uint64_t grids[4];
        uint32_t signs = 0;
        if (type == GGML_TYPE_IQ2_S) {
            const auto * b = static_cast<const block_iq2_s *>(packed);
            for (int l = 0; l < 4; ++l) { grids[l] = iq2s_grid[b->qs[4 * g + l] | ((b->qh[g] << (8 - 2 * l)) & 0x300)]; }
            std::memcpy(&signs, b->qs + QK_K / 8 + 4 * g, sizeof(signs));
        } else if (type == GGML_TYPE_IQ2_XS) {
            const auto * b = static_cast<const block_iq2_xs *>(packed);
            for (int l = 0; l < 4; ++l) {
                const uint16_t q = b->qs[4 * g + l];
                grids[l] = iq2xs_grid[q & 511];
                signs |= uint32_t(ksigns_iq2xs[q >> 9]) << (8 * l);
            }
        } else if (type == GGML_TYPE_IQ3_S) {
            const auto * b = static_cast<const block_iq3_s *>(packed);
            for (int l = 0; l < 4; ++l) {
                const uint32_t lo = iq3s_grid[b->qs[8 * g + 2 * l] | ((b->qh[g] << (8 - 2 * l)) & 256)];
                const uint32_t hi = iq3s_grid[b->qs[8 * g + 2 * l + 1] | ((b->qh[g] << (7 - 2 * l)) & 256)];
                grids[l] = uint64_t(lo) | (uint64_t(hi) << 32);
            }
            std::memcpy(&signs, b->signs + 4 * g, sizeof(signs));
        } else {
            const auto * b = static_cast<const block_iq3_xxs *>(packed);
            uint32_t aux; std::memcpy(&aux, b->qs + QK_K / 4 + 4 * g, sizeof(aux));
            for (int l = 0; l < 4; ++l) {
                const uint32_t lo = iq3xxs_grid[b->qs[8 * g + 2 * l]];
                const uint32_t hi = iq3xxs_grid[b->qs[8 * g + 2 * l + 1]];
                grids[l] = uint64_t(lo) | (uint64_t(hi) << 32);
                signs |= uint32_t(ksigns_iq2xs[(aux >> (7 * l)) & 127]) << (8 * l);
            }
        }
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(grids));
        const __m256i sign_bytes = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_cvtsi32_si128(signs)),
            _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
        const __m256i mask = _mm256_set1_epi64x(int64_t(0x8040201008040201ull));
        const __m256i negative = _mm256_cmpeq_epi8(_mm256_and_si256(sign_bytes, mask), mask);
        return _mm256_sub_epi8(_mm256_xor_si256(v, negative), negative);
    }
}

static void ggml_cpu_tier_dot32(__m256i w, const int8_t * codes, bool halves, int * dots) {
    const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(codes));
    // Codes are in [-127, 127]; even the IQ4 pair sums fit signed 16 bits.
    const __m256i pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(q, w));
    const __m256i sums = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    int v[8]; _mm256_storeu_si256(reinterpret_cast<__m256i *>(v), sums);
    if (halves) {
        dots[0] = v[0] + v[1] + v[2] + v[3];
        dots[1] = v[4] + v[5] + v[6] + v[7];
    } else {
        dots[0] = v[0] + v[1] + v[4] + v[5];
        dots[1] = v[2] + v[3] + v[6] + v[7];
    }
}

static void ggml_cpu_tier_quantize32(const float * x, int8_t * codes, float & scale, const ggml_cpu_tier_quant_policy & config) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 amax8 = _mm256_setzero_ps();
    for (int i = 0; i < 32; i += 8) { amax8 = _mm256_max_ps(amax8, _mm256_andnot_ps(sign, _mm256_loadu_ps(x + i))); }
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(amax8), _mm256_extractf128_ps(amax8, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    const float amax = _mm_cvtss_f32(m);
    const float d = config.reciprocal ? amax * (1.0f / 127.0f) : amax / 127.0f;
    const float inverse = amax ? (config.reciprocal ? 127.0f * ggml_cpu_tier_reciprocal(amax, config.reciprocal) : 127.0f / amax) : 0;
    scale = config.kind == 1 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(d)) :
        (amax ? (config.reciprocal ? ggml_cpu_tier_reciprocal(inverse, config.reciprocal) : 1.0f / inverse) : 0);
    const float factor = config.kind == 1 && config.reciprocal ? ggml_cpu_tier_reciprocal(d, config.reciprocal) : inverse;
    for (int i = 0; i < 32; i += 8) {
        const __m256 input = _mm256_loadu_ps(x + i);
        __m256 v = config.kind == 1 && !config.reciprocal ? _mm256_div_ps(input, _mm256_set1_ps(d)) : _mm256_mul_ps(input, _mm256_set1_ps(factor));
        if (!amax) { v = _mm256_setzero_ps(); }
        const __m256 truncated = _mm256_round_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        const __m256 fraction = _mm256_andnot_ps(sign, _mm256_sub_ps(v, truncated));
        const __m256 adjust = _mm256_and_ps(_mm256_cmp_ps(fraction, _mm256_set1_ps(0.5f), _CMP_GE_OQ), _mm256_or_ps(_mm256_and_ps(v, sign), _mm256_set1_ps(1.0f)));
        const __m256i rounded = _mm256_cvttps_epi32(_mm256_add_ps(truncated, adjust));
        const __m128i packed16 = _mm_packs_epi32(_mm256_castsi256_si128(rounded), _mm256_extracti128_si256(rounded, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i *>(codes + i), _mm_packs_epi16(packed16, packed16));
    }
}
#endif
#if defined(__AVX2__) && defined(__FMA__)
template<ggml_type type, int policy>
static void ggml_cpu_tier_quant_matmul_avx2(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto & config = *static_cast<const ggml_cpu_tier_quant_policy *>(userdata);
    GGML_ASSERT(policy == 1 || policy == 2);
    const int lanes = config.mmvq_warps == 4 ? 128 : 32;
    const auto * weight = dst->src[0];
    GGML_ASSERT(type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS ||
                type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ4_NL ||
                type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_Q2_0);
    const auto * input = dst->src[1];
    const auto * ids = dst->src[2];
    const auto * traits = ggml_get_type_traits(type);
    const int64_t width = weight->ne[0], rows = weight->ne[1], block = type == GGML_TYPE_IQ4_NL ? 32 : type == GGML_TYPE_Q2_0 ? 64 : 256;
    GGML_ASSERT(width % 32 == 0 && width % block == 0 && block <= 256);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && input->ne[0] == width && input->ne[1] == 1);
    GGML_ASSERT(input->ne[3] == 1 && ids->ne[2] == 1 && ids->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[0] == 1 && ids->ne[1] == input->ne[2]);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->ne[0] == rows && dst->ne[1] == 1 && dst->ne[2] == input->ne[2]);
    thread_local std::vector<int8_t> codes;
    thread_local std::vector<float> scales;
    codes.resize(width); scales.resize(width / 32);
    const int64_t jobs = input->ne[2] * rows;
    const int64_t begin = jobs * ith / nth, end = jobs * (ith + 1) / nth;
    int64_t previous_pair = -1;
    for (int64_t job = begin; job < end; ++job) {
        const int64_t pair = job / rows, row = job % rows;
        if (pair != previous_pair) {
            const auto * x = reinterpret_cast<const float *>(static_cast<const char *>(input->data) + pair * input->nb[2]);
            for (int64_t c = 0; c < width; c += 32) { ggml_cpu_tier_quantize32(x + c, codes.data() + c, scales[c / 32], config); }
            previous_pair = pair;
        }
        int32_t expert;
        std::memcpy(&expert, static_cast<const char *>(ids->data) + pair * ids->nb[1], sizeof(expert));
        GGML_ASSERT(expert >= 0 && expert < weight->ne[2]);
        const auto * raw = static_cast<const char *>(weight->data) + expert * weight->nb[2] + row * weight->nb[1];
        float sum = 0;
        std::array<float, 128> partial{};
        for (int64_t c = 0; c < width; c += block) {
            const auto * packed = raw + (c / block) * traits->type_size;
            ggml_fp16_t half;
            std::memcpy(&half, packed, sizeof(half));
            float d = ggml_fp16_to_fp32(half);
            if (type == GGML_TYPE_IQ3_XXS) { d *= 0.25f; }
            if (d == 0) { continue; }
            if (policy == 1 && block == 256) {
                int integers[8];
                for (int g = 0; g < 8; ++g) {
                    const __m256i values = ggml_cpu_tier_decode32<type>(packed, g);
                    int dots[2];
                    ggml_cpu_tier_dot32(values, codes.data() + c + 32 * g, (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XS), dots);
                    if (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XS) {
                        const auto * scales2 = type == GGML_TYPE_IQ2_S ? reinterpret_cast<const block_iq2_s *>(packed)->scales : reinterpret_cast<const block_iq2_xs *>(packed)->scales;
                        const int ls0 = scales2[g] & 15, ls1 = scales2[g] >> 4;
                        integers[g] = (dots[0] * ls0 + dots[1] * ls1 + (dots[0] + dots[1]) / 2) / 4;
                    } else if (type == GGML_TYPE_IQ3_XXS) {
                        const auto * b = reinterpret_cast<const block_iq3_xxs *>(packed);
                        uint32_t aux; std::memcpy(&aux, b->qs + QK_K / 4 + 4 * g, sizeof(aux));
                        const int dot = dots[0] + dots[1];
                        integers[g] = (int(aux >> 28) * dot + dot / 2) / 2;
                    } else {
                        int multiplier;
                        if (type == GGML_TYPE_IQ3_S) {
                            const auto * b = reinterpret_cast<const block_iq3_s *>(packed);
                            multiplier = 1 + 2 * ((b->scales[g / 2] >> (4 * (g % 2))) & 15);
                        } else {
                            const auto * b = reinterpret_cast<const block_iq4_xs *>(packed);
                            multiplier = (((b->scales_l[g / 2] >> (4 * (g % 2))) & 15) | (((b->scales_h >> (2 * g)) & 3) << 4)) - 32;
                        }
                        integers[g] = (dots[0] + dots[1]) * multiplier;
                    }
                }
                const __m256 scaled = _mm256_mul_ps(_mm256_set1_ps(type == GGML_TYPE_IQ3_XXS ? 4 * d : d), _mm256_loadu_ps(scales.data() + c / 32));
                const __m256 dot = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(integers)));
                float * lanes8 = partial.data() + ((c / 32) & (lanes - 1));
                _mm256_storeu_ps(lanes8, _mm256_fmadd_ps(scaled, dot, _mm256_loadu_ps(lanes8)));
                continue;
            }
            for (int g = 0; g < block / 32; ++g) {
                const __m256i values = ggml_cpu_tier_decode32<type>(packed, g);
                int dots[2];
                ggml_cpu_tier_dot32(values, codes.data() + c + 32 * g, (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XS), dots);
                const float activation_scale = scales[c / 32 + g];
                const int lane = (c / 32 + g) & (lanes - 1);
                if (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XS) {
                    const auto * scales2 = type == GGML_TYPE_IQ2_S ? reinterpret_cast<const block_iq2_s *>(packed)->scales : reinterpret_cast<const block_iq2_xs *>(packed)->scales;
                    const int ls0 = scales2[g] & 15, ls1 = scales2[g] >> 4;
                    const float s0 = (ls0 * d + d / 2) / 4, s1 = (ls1 * d + d / 2) / 4;
                    if (policy == 2) { sum = std::fma(std::fma(float(dots[0]), s0, float(dots[1]) * s1), activation_scale, sum); }
                    else {
                        const int dot = (dots[0] * ls0 + dots[1] * ls1 + (dots[0] + dots[1]) / 2) / 4;
                        partial[lane] = std::fma(d * activation_scale, float(dot), partial[lane]);
                    }
                    continue;
                }
                int multiplier = 1;
                if (type == GGML_TYPE_IQ3_S) {
                    const auto * b = reinterpret_cast<const block_iq3_s *>(packed);
                    multiplier = 1 + 2 * ((b->scales[g / 2] >> (4 * (g % 2))) & 15);
                } else if (type == GGML_TYPE_IQ4_XS) {
                    const auto * b = reinterpret_cast<const block_iq4_xs *>(packed);
                    multiplier = (((b->scales_l[g / 2] >> (4 * (g % 2))) & 15) | (((b->scales_h >> (2 * g)) & 3) << 4)) - 32;
                } else if (type == GGML_TYPE_IQ3_XXS) {
                    const auto * b = reinterpret_cast<const block_iq3_xxs *>(packed);
                    uint32_t scale_signs;
                    std::memcpy(&scale_signs, b->qs + QK_K / 4 + 4 * g, sizeof(scale_signs));
                    multiplier = 1 + 2 * (scale_signs >> 28);
                }
                const float subscale = d * multiplier;
                if (subscale == 0) { continue; }
                if (policy == 2) { sum = std::fma(float(dots[0] + dots[1]) * subscale, activation_scale, sum); }
                else if (type == GGML_TYPE_IQ3_XXS) {
                    const int dot = dots[0] + dots[1], ls = (multiplier - 1) / 2;
                    const int scaled = (ls * dot + dot / 2) / 2;
                    partial[lane] = std::fma((4 * d) * activation_scale, float(scaled), partial[lane]);
                } else if (type == GGML_TYPE_IQ4_NL) {
                    for (int i = 0; i < 2; ++i) { partial[((c + 32 * g) / 16 + i) & (lanes - 1)] = std::fma(d * activation_scale, float(dots[i]), partial[((c + 32 * g) / 16 + i) & (lanes - 1)]); }
                } else { partial[lane] = std::fma(d * activation_scale, float((dots[0] + dots[1]) * multiplier), partial[lane]); }
            }
        }
        if (policy == 1) {
            for (int warp = 1; warp < lanes / 32; ++warp) {
                for (int i = 0; i < 32; ++i) { partial[i] += partial[warp * 32 + i]; }
            }
            for (int offset = 16; offset; offset /= 2) {
                for (int i = 0; i < offset; ++i) { partial[i] += partial[i + offset]; }
            }
            sum = partial[0];
        }
        auto * out = reinterpret_cast<float *>(static_cast<char *>(dst->data) + pair * dst->nb[2]);
        out[row] = sum;
    }
}
#endif

void ggml_cpu_tier_quant_matmul(ggml_tensor * dst, int ith, int nth, void * userdata) {
#if defined(__AVX2__) && defined(__FMA__)
    if (!std::getenv("CNM_TIER_CUDA_SCALAR")) {
        const auto & config = *static_cast<const ggml_cpu_tier_quant_policy *>(userdata);
        GGML_ASSERT(config.kind == 1 || config.kind == 2);
#define GGML_TIER_CASE(type) case type: \
        if (config.kind == 1) { ggml_cpu_tier_quant_matmul_avx2<type, 1>(dst, ith, nth, userdata); } \
        else { ggml_cpu_tier_quant_matmul_avx2<type, 2>(dst, ith, nth, userdata); } return;
        switch (dst->src[0]->type) {
            GGML_TIER_CASE(GGML_TYPE_IQ3_S)
            GGML_TIER_CASE(GGML_TYPE_IQ3_XXS)
            GGML_TIER_CASE(GGML_TYPE_IQ2_S)
            GGML_TIER_CASE(GGML_TYPE_IQ2_XS)
            GGML_TIER_CASE(GGML_TYPE_IQ4_XS)
            GGML_TIER_CASE(GGML_TYPE_IQ4_NL)
            GGML_TIER_CASE(GGML_TYPE_Q2_0)
            default: GGML_ABORT("unsupported CUDA arithmetic weight type");
        }
#undef GGML_TIER_CASE
    }
#endif
    ggml_cpu_tier_quant_matmul_scalar(dst, ith, nth, userdata);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        if (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        if (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        if (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        if (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        return bufts;
    }();

    return bufts;
}

static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    static std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        return bufts;
    }();

    return extra_bufts.data();

    GGML_UNUSED(device);
}

static bool ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    for (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra == buft) {
            return true;
        }
    }
    return false;
}

// CPU backend - backend (stream)

struct ggml_backend_cpu_context {
    int                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    bool                use_ref;  // use reference implementation
};

static const char * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    return "CPU";

    GGML_UNUSED(backend);
}

static void ggml_backend_cpu_free(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

struct ggml_backend_plan_cpu {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

static ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_plan_cpu * cpu_plan = new ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = new uint8_t[cpu_plan->cplan.work_size];
        if (cpu_plan->cplan.work_data == NULL) {
            delete cpu_plan;
            return NULL;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    return cpu_plan;
}

static void ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    return ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    if (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = new uint8_t[cplan.work_size];
        if (cpu_ctx->work_data == NULL) {
            cpu_ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    return ggml_graph_compute(cgraph, &cplan);
}

static const struct ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_cpu_guid(void) {
    static ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    return &guid;
}

ggml_backend_t ggml_backend_cpu_init(void) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    struct ggml_backend_cpu_context * ctx = new ggml_backend_cpu_context;
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = NULL;
    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;
    ctx->use_ref             = false;

    ggml_backend_t cpu_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    if (cpu_backend == NULL) {
        delete ctx;
        return NULL;
    }

    return cpu_backend;
}

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;

    if (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

struct ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        if (!sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, NULL, 0); // NOLINT
        }
#elif defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) {
                if (strncmp(buf, "model name", 10) == 0) {
                    char * p = strchr(buf, ':');
                    if (p) {
                        p++;
                        while (std::isspace(*p)) {
                            p++;
                        }
                        while (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        break;
                    }
                }
            }
            fclose(f);
        }
#elif defined(_WIN32)
        HKEY hKey;
        if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        KEY_READ,
                        &hKey) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            if (RegQueryValueExA(hKey,
                                "ProcessorNameString",
                                NULL,
                                NULL,
                                NULL,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                if (RegQueryValueExA(hKey,
                                    "ProcessorNameString",
                                    NULL,
                                    NULL,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    if (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            RegCloseKey(hKey);
        }
#endif
    }
};

static const char * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    return "CPU";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free = status.ullAvailPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *free = *total;
#endif // _WIN32

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        return true;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    for (int i = 0; i < 4; i++) {
        if (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            return buf_extra->supports_op(dev, op);
        }
    }

    switch (op->op) {
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
            return
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        case GGML_OP_MUL_MAT:
            if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                return src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            return src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        case GGML_OP_SOFT_MAX_BACK: {
            if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                return false;
            }
            float max_bias = 0.0f;

            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));

            return max_bias == 0.0f;
        }
        case GGML_OP_IM2COL_BACK:
            return src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        case GGML_OP_GET_ROWS_BACK:
            return src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        case GGML_OP_OUT_PROD:
            return (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_2D:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_SSM_SCAN:
            return ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        default:
            return true;
    }
}

static bool ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// CPU backend - backend (reg)

static const char * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    return "CPU";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_cpu_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_cpu_device_context ctx;
    static ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    return &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
static ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        if (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        if (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        if (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        if (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        if (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        if (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        if (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        if (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        if (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        if (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        if (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        if (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        if (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        if (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        if (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        if (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        if (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        if (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        if (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        if (ggml_cpu_get_sve_cnt() > 0) {
            static std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        if (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        if (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        if (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        if (ggml_cpu_get_rvv_vlen() > 0) {
            static std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        if (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        if (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        if (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        if (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #ifdef GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #endif
    #ifdef GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #endif
    #ifdef GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #endif
    #ifdef GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #endif
    #ifdef GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #endif

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cpu_get_features;
    }
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_cpu_set_abort_callback;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        return (void *)ggml_numa_init;
    }
    if (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        return (void *)ggml_is_numa;
    }
    if (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        return (void *)ggml_backend_cpu_set_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    if (strcmp(name, "ggml_threadpool_new") == 0) {
        return (void *)ggml_threadpool_new;
    }
    if (strcmp(name, "ggml_threadpool_free") == 0) {
        return (void *)ggml_threadpool_free;
    }
    if (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        return (void *)ggml_backend_cpu_set_threadpool;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    // init CPU feature detection
    ggml_cpu_init();

    static struct ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
