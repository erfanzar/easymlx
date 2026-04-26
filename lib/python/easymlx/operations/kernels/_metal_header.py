# Copyright 2026 The EASYDEL / EASYMLX Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared Metal header for paged attention kernels.

This header provides bfloat16 polyfill, FP8 helpers, vector types,
dot-product utilities, and cache conversion helpers needed by all
paged attention Metal kernels.
"""

PAGED_ATTENTION_HEADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ===== bfloat16 polyfill (from utils.metal) =====


#if defined(__HAVE_BFLOAT__)

typedef bfloat bfloat16_t;

#else

/////////////////////////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////////////////////////

constexpr METAL_FUNC uint16_t float_to_bfloat_bits(float x) {
  // Check for nan
  if ((as_type<uint32_t>(x) & ~_fp_encoding_traits<float>::sign_mask) >
      _fp_encoding_traits<float>::inf_mask) {
    return uint16_t(as_type<uint32_t>(0x7FC0));
  }
  // Take bits
  uint32_t float_bits = as_type<uint32_t>(x);

  // Round to nearest even
  float_bits += ((float_bits >> 16) & 1) + as_type<uint32_t>(0x7FFF);

  // Take upper 16 bits
  return float_bits >> 16;
}

constexpr METAL_FUNC float bfloat_bits_to_float(uint16_t x) {
  // Upper 16 bits are the data and lower 16 bits are 0s
  return as_type<float>((uint32_t)x << 16);
}

struct _MLX_BFloat16;

template <typename T>
static constexpr constant bool can_convert_to_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<float, T>;

/////////////////////////////////////////////////////////////////////////////
// Bfloat struct
/////////////////////////////////////////////////////////////////////////////

struct _MLX_BFloat16 {
  /////////////////////////////////////////////////////////////////////////////
  // Constructors
  uint16_t bits_;
  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;

  struct bits_to_bfloat_struct {};
  static constexpr METAL_FUNC bits_to_bfloat_struct bits_to_bfloat() {
    return bits_to_bfloat_struct();
  }
  constexpr METAL_FUNC _MLX_BFloat16(uint16_t bits, bits_to_bfloat_struct)
      : bits_(bits) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions to bfloat

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) thread
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) threadgroup
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) device
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) constant
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions from bfloat

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const thread {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const threadgroup {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const device {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() constant {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
};

/////////////////////////////////////////////////////////////////////////////
// Bfloat operators
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Unary ops
constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 x) {
  return -static_cast<float>(x);
}

/////////////////////////////////////////////////////////////////////////////
// Binary operators
#define bfloat_binop_base(__op__, __operator__, otype, atype, btype, ctype)    \
  constexpr METAL_FUNC otype __operator__(atype lhs, btype rhs) {              \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \
  }

#define bfloat_binop_helper(__op__, __operator__, otype, itype, ctype)         \
  constexpr METAL_FUNC otype __operator__(_MLX_BFloat16 lhs, itype rhs) {      \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \
  }                                                                            \
  constexpr METAL_FUNC otype __operator__(itype lhs, _MLX_BFloat16 rhs) {      \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \
  }

/////////////////////////////////////////////////////////////////////////////
// Arithmetic Operators
#define bfloat_binop(_op_, _operator_)                                         \
  bfloat_binop_base(_op_, _operator_, _MLX_BFloat16, _MLX_BFloat16,            \
                    _MLX_BFloat16, float);                                     \
  bfloat_binop_helper(_op_, _operator_, float, float, float);                  \
  bfloat_binop_helper(_op_, _operator_, float, half, float);                   \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int32_t, float);        \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint32_t, float);       \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int64_t, float);        \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint64_t, float);

bfloat_binop(+, operator+);
bfloat_binop(-, operator-);
bfloat_binop(*, operator*);
bfloat_binop(/, operator/);

/////////////////////////////////////////////////////////////////////////////
// Comparison ops
#define bfloat_compop(__op__, __operator__)                                    \
  bfloat_binop_base(__op__, __operator__, bool, _MLX_BFloat16, _MLX_BFloat16,  \
                    float);                                                    \
  bfloat_binop_helper(__op__, __operator__, bool, float, float);               \
  bfloat_binop_helper(__op__, __operator__, bool, half, float);                \
  bfloat_binop_helper(__op__, __operator__, bool, int32_t, float);             \
  bfloat_binop_helper(__op__, __operator__, bool, uint32_t, float);            \
  bfloat_binop_helper(__op__, __operator__, bool, int64_t, float);             \
  bfloat_binop_helper(__op__, __operator__, bool, uint64_t, float);

bfloat_compop(>, operator>);
bfloat_compop(<, operator<);
bfloat_compop(>=, operator>=);
bfloat_compop(<=, operator<=);
bfloat_compop(==, operator==);
bfloat_compop(!=, operator!=);

#undef bfloat_compop
#undef bfloat_binop_base
#undef bfloat_binop_helper
#undef bfloat_binop

/////////////////////////////////////////////////////////////////////////////
// Inplace Operators
#define bfloat_inplace_op_helper(__op__, __operator__, itype, addr_space)      \
  constexpr METAL_FUNC addr_space _MLX_BFloat16 &__operator__(                 \
      addr_space _MLX_BFloat16 &lhs, itype rhs) {                              \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);              \
    return lhs;                                                                \
  }                                                                            \
  constexpr METAL_FUNC addr_space itype &__operator__(addr_space itype &lhs,   \
                                                      _MLX_BFloat16 rhs) {     \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);              \
    return lhs;                                                                \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__, itype)       \
  bfloat_inplace_op_helper(__op__, __operator__, itype, device);               \
  bfloat_inplace_op_helper(__op__, __operator__, itype, thread);               \
  bfloat_inplace_op_helper(__op__, __operator__, itype, threadgroup);

#define bfloat_inplace_op(itype)                                               \
  bfloat_inplace_op_addr_space_helper(+, operator+=, itype);                   \
  bfloat_inplace_op_addr_space_helper(-, operator-=, itype);                   \
  bfloat_inplace_op_addr_space_helper(*, operator*=, itype);                   \
  bfloat_inplace_op_addr_space_helper(/, operator/=, itype);

bfloat_inplace_op(float);
bfloat_inplace_op(half);
bfloat_inplace_op(int16_t);
bfloat_inplace_op(int32_t);
bfloat_inplace_op(int64_t);
bfloat_inplace_op(uint16_t);
bfloat_inplace_op(uint32_t);
bfloat_inplace_op(uint64_t);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper
#undef bfloat_inplace_op

#define bfloat_inplace_op_helper(__op__, __operator__, addr_space)             \
  constexpr METAL_FUNC addr_space _MLX_BFloat16 &__operator__(                 \
      addr_space _MLX_BFloat16 &lhs, _MLX_BFloat16 rhs) {                      \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);              \
    return lhs;                                                                \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__)              \
  bfloat_inplace_op_helper(__op__, __operator__, device);                      \
  bfloat_inplace_op_helper(__op__, __operator__, thread);                      \
  bfloat_inplace_op_helper(__op__, __operator__, threadgroup);

bfloat_inplace_op_addr_space_helper(+, operator+=);
bfloat_inplace_op_addr_space_helper(-, operator-=);
bfloat_inplace_op_addr_space_helper(*, operator*=);
bfloat_inplace_op_addr_space_helper(/, operator/=);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper

/////////////////////////////////////////////////////////////////////////////
// Bfloat typedef
/////////////////////////////////////////////////////////////////////////////

typedef struct _MLX_BFloat16 bfloat16_t;

#endif


// ===== FP8 helpers (from float8.metal) =====


// Helpers ------------------------------------------------------------
static inline uint as_bits(float x) { return as_type<uint>(x); }
static inline float from_bits(uint b) { return as_type<float>(b); }

// -------------------------------------------------------------------
// FP8 E4M3 (bias = 7)
// -------------------------------------------------------------------
inline float fp8_e4m3_to_float(uchar v) {
  const uint s = v >> 7;
  const uint exp = (v >> 3) & 0xF;
  const uint man = v & 0x7;

  if (exp == 0) { // zero / sub-normal
    if (man == 0)
      return s ? -0.f : 0.f;
    const float m = float(man) / 8.f; // already scaled by 2^-3
    float val = ldexp(m, 1 - 7);      // 2^(1-bias) = 2^-6
    return s ? -val : val;
  }

  // E4M3 has NO infinity - only NaN when exp=15 and mantissa=7
  if (exp == 0xF && man == 0x7) {
    return NAN;
  }

  // Normalized (including exp=0xF with mantissa 0-6, which are valid numbers)
  const float m = 1.f + float(man) / 8.f;
  float val = ldexp(m, int(exp) - 7);
  return s ? -val : val;
}

// -------------------------------------------------------------------
// FP8 E5M2 (bias = 15)
// -------------------------------------------------------------------
inline float fp8_e5m2_to_float(uchar v) {
  const uint s = v >> 7;
  const uint exp = (v >> 2) & 0x1F;
  const uint man = v & 0x3;

  if (exp == 0) {
    if (man == 0)
      return s ? -0.f : 0.f;
    const float m = float(man) / 4.f;
    float val = ldexp(m, 1 - 15); // 2^(1-bias) = 2^-14
    return s ? -val : val;
  }

  if (exp == 0x1F) {
    if (man != 0)
      return NAN;
    return s ? -INFINITY : INFINITY;
  }

  const float m = 1.f + float(man) / 4.f;
  float val = ldexp(m, int(exp) - 15);
  return s ? -val : val;
}

// -------------------------------------------------------------------
// Encoding helpers (round-to-nearest-even, gradual under-flow, sat-to-∞)
// -------------------------------------------------------------------
namespace detail {
template <int EXP_BITS, int MAN_BITS, int BIAS>
inline uchar fp32_to_fp8(float f) {
  const uint bits = as_bits(f);
  const uint s = bits >> 31;
  const uint abs = bits & 0x7FFFFFFF;

  // NaN propagates, Inf saturates
  if (abs >= 0x7F800000u) {
    return uchar((s << 7) | (((1u << EXP_BITS) - 1u) << MAN_BITS) |
                 (abs != 0x7F800000u));
  }

  int e = int((abs >> 23) & 0xFF) - 127;   // unbiased exponent
  uint m = abs & 0x7FFFFFu;                // 23-bit mantissa
  const int EXP_MAX = (1 << EXP_BITS) - 2; // last finite exponent

  // ---------- Normal path -------------------------------------------------
  int e_fp8 = e + BIAS;
  if (e_fp8 >= 1 && e_fp8 <= EXP_MAX) {
    // round-to-nearest-even
    const int shift = 23 - MAN_BITS;
    uint mant = m >> shift;
    const uint lsb = mant & 1u;
    const uint round = (m >> (shift - 1)) & 1u;
    const uint sticky = (m & ((1u << (shift - 1)) - 1u)) != 0u;
    mant += (round & (sticky | lsb));
    if (mant >> MAN_BITS) { // mantissa overflow
      mant = 0;
      ++e_fp8;
      if (e_fp8 > EXP_MAX)
        return uchar((s << 7) | (((1u << EXP_BITS) - 1u) << MAN_BITS)); // ∞
    }
    return uchar((s << 7) | (uint(e_fp8) << MAN_BITS) |
                 (mant & ((1u << MAN_BITS) - 1u)));
  }

  // ---------- Sub-normal / under-flow ------------------------------------
  if (e_fp8 < 1 - MAN_BITS) // too small -> ±0
    return uchar(s << 7);

  // shift so that exponent becomes 1
  int rshift = (1 - e_fp8) + (23 - MAN_BITS);
  uint mant = (0x800000u | m); // implicit 1
  uint rounded = (mant + (1u << (rshift - 1))) >> rshift;
  if (rounded == 0)
    return uchar(s << 7); // rounds to zero

  return uchar((s << 7) | (rounded & ((1u << MAN_BITS) - 1u)));
}
} // namespace detail

inline uchar float_to_fp8_e4m3(float f) {
  // E4M3 has no infinity - must handle specially
  // Max value is 448 (exp=15, mantissa=6), mantissa=7 is NaN

  if (isnan(f)) {
    return 0x7F; // positive NaN (exp=15, mantissa=7)
  }

  const uint bits = as_bits(f);
  const uint s = bits >> 31;

  // Clamp infinity and overflow to max value (448)
  if (isinf(f) || fabs(f) > 448.0f) {
    // E4M3 max: exp=15, mantissa=6 (value = 1.75 * 2^8 = 448)
    return uchar((s << 7) | (0xF << 3) | 0x6);
  }

  // Use the template for normal values, but check result
  uchar result = detail::fp32_to_fp8<4, 3, 7>(f);

  // Ensure we don't accidentally create NaN or invalid encoding
  uint exp_bits = (result >> 3) & 0xF;
  uint man_bits = result & 0x7;
  if (exp_bits == 0xF && man_bits == 0x7) {
    // Would be NaN, clamp to max value instead
    return uchar((s << 7) | (0xF << 3) | 0x6);
  }

  return result;
}
inline uchar float_to_fp8_e5m2(float f) {
  return detail::fp32_to_fp8<5, 2, 15>(f);
}


// ===== Vector types & dot product (from pagedattention.metal header) =====


// ========================================== Generic vector types

// A vector type to store Q, K, V elements.
template <typename T, int VEC_SIZE> struct Vec {};

// A vector type to store FP32 accumulators.
template <typename T> struct FloatVec {};

// Template vector operations.
template <typename Acc, typename A, typename B> inline Acc mul(A a, B b);

template <typename T> inline float sum(T v);

template <typename T> inline float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

template <typename A, typename T> inline float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

// FP32 vector data types.
struct Float8_ {
  float4 x;
  float4 y;
};

template <> struct Vec<float, 1> {
  using Type = float;
};
template <> struct Vec<float, 2> {
  using Type = float2;
};
template <> struct Vec<float, 4> {
  using Type = float4;
};
template <> struct Vec<float, 8> {
  using Type = Float8_;
};

template <> struct FloatVec<float> {
  using Type = float;
};
template <> struct FloatVec<float2> {
  using Type = float2;
};
template <> struct FloatVec<float4> {
  using Type = float4;
};
template <> struct FloatVec<Float8_> {
  using Type = Float8_;
};

template <> inline float mul(float a, float b) { return a * b; }

template <> inline float2 mul(float2 a, float2 b) { return a * b; }

template <> inline float4 mul(float4 a, float4 b) { return a * b; }

template <> inline Float8_ mul(Float8_ a, Float8_ b) {
  Float8_ c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <> inline float sum(float a) { return a; }

template <> inline float sum(float2 a) { return a.x + a.y; }

template <> inline float sum(float4 a) { return a.x + a.y + a.z + a.w; }

template <> inline float sum(Float8_ a) { return sum(a.x) + sum(a.y); }

inline Float8_ fma(Float8_ a, Float8_ b, Float8_ c) {
  Float8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread float &dst, float src) { dst = src; }
inline void from_float(thread float2 &dst, float2 src) { dst = src; }
inline void from_float(thread float4 &dst, float4 src) { dst = src; }
inline void from_float(thread Float8_ &dst, Float8_ src) { dst = src; }

// BF16 vector data types.
// #if defined(__HAVE_BFLOAT__)

// struct Bfloat8_ {
//   bfloat4 x;
//   bfloat4 y;
// };

// template<>
// struct Vec<bfloat, 1> {
//   using Type = bfloat;
// };
// template<>
// struct Vec<bfloat, 2> {
//   using Type = bfloat2;
// };
// template<>
// struct Vec<bfloat, 4> {
//   using Type = bfloat4;
// };
// template<>
// struct Vec<bfloat, 8> {
//   using Type = Bfloat8_;
// };

// template<>
// struct FloatVec<bfloat> {
//   using Type = float;
// };
// template<>
// struct FloatVec<bfloat2> {
//   using Type = float2;
// };
// template<>
// struct FloatVec<bfloat4> {
//   using Type = float4;
// };
// template<>
// struct FloatVec<Bfloat8_> {
//   using Type = Float8_;
// };

// template<>
// inline float mul(bfloat a, bfloat b) {
//   return (float)a * (float)b;
// }
// template<>
// inline bfloat mul(bfloat a, bfloat b) {
//   return a*b;
// }

// template<>
// inline float2 mul(bfloat2 a, bfloat2 b) {
//   return (float2)a * (float2)b;
// }
// template<>
// inline bfloat2 mul(bfloat2 a, bfloat2 b) {
//   return a * b;
// }

// template<>
// inline float4 mul(bfloat4 a, bfloat4 b) {
//   return (float4)a * (float4)b;
// }
// template<>
// inline bfloat4 mul(bfloat4 a, bfloat4 b) {
//   return a * b;
// }

// template<>
// inline Float8_ mul(Bfloat8_ a, Bfloat8_ b) {
//   Float8_ c;
//   c.x = mul<float4, bfloat4, bfloat4>(a.x, b.x);
//   c.y = mul<float4, bfloat4, bfloat4>(a.y, b.y);
//   return c;
// }
// template<>
// inline Bfloat8_ mul(Bfloat8_ a, Bfloat8_ b) {
//   Bfloat8_ c;
//   c.x = mul<bfloat4, bfloat4, bfloat4>(a.x, b.x);
//   c.y = mul<bfloat4, bfloat4, bfloat4>(a.y, b.y);
//   return c;
// }

// template<>
// inline float sum(bfloat a) {
//   return (float)a;
// }

// template<>
// inline float sum(bfloat2 a) {
//   return (float)a.x + (float)a.y;
// }

// template<>
// inline float sum(bfloat4 a) {
//   return sum(a.x) + sum(a.y);
// }

// template<>
// inline float sum(Bfloat8_ a) {
//   return sum(a.x) + sum(a.y);
// }

// inline float fma(bfloat a, bfloat b, float c) {
//   return (float)a * (float)b + c;
// }

// inline float2 fma(bfloat2 a, bfloat2 b, float2 c) {
//   return (float2)a * (float2)b + c;
// }

// inline float4 fma(bfloat4 a, bfloat4 b, float4 c) {
//   return (float4)a * (float4)b + c;
// }

// inline Float8_ fma(Bfloat8_ a, Bfloat8_ b, Float8_ c) {
//   Float8_ res;
//   res.x = fma((float4)a.x, (float4)b.x, (float4)c.x);
//   res.y = fma((float4)a.y, (float4)b.y, (float4)c.y);
//   return res;
// }
// inline Bfloat8_ fma(Bfloat8_ a, Bfloat8_ b, Bfloat8_ c) {
//   Bfloat8_ res;
//   res.x = (bfloat4)fma((float4)a.x, (float4)b.x, (float4)c.x);
//   res.y = (bfloat4)fma((float4)a.y, (float4)b.x, (float4)c.y);
//   return c;
// }

// inline void from_float(thread bfloat& dst, float src) {
//   dst = static_cast<bfloat>(src);
// }
// inline void from_float(thread bfloat2& dst, float2 src) {
//   dst.x = static_cast<bfloat>(src.x);
//   dst.y = static_cast<bfloat>(src.y);
// }
// inline void from_float(thread bfloat4& dst, float4 src) {
//   dst.x = static_cast<bfloat>(src.x);
//   dst.y = static_cast<bfloat>(src.y);
//   dst.z = static_cast<bfloat>(src.z);
//   dst.w = static_cast<bfloat>(src.w);
// }
// inline void from_float(thread Bfloat8_& dst, Float8_ src) {
//   bfloat4 x;
//   bfloat4 y;
//   from_float(x, src.x);
//   from_float(y, src.y);
//   dst.x = x;
//   dst.y = y;
// }

// #else

struct Bfloat2_ {
  bfloat16_t x;
  bfloat16_t y;
};

struct Bfloat4_ {
  Bfloat2_ x;
  Bfloat2_ y;
};

struct Bfloat8_ {
  Bfloat4_ x;
  Bfloat4_ y;
};

template <> struct Vec<bfloat16_t, 1> {
  using Type = bfloat16_t;
};
template <> struct Vec<bfloat16_t, 2> {
  using Type = Bfloat2_;
};
template <> struct Vec<bfloat16_t, 4> {
  using Type = Bfloat4_;
};
template <> struct Vec<bfloat16_t, 8> {
  using Type = Bfloat8_;
};

template <> struct FloatVec<bfloat16_t> {
  using Type = float;
};
template <> struct FloatVec<Bfloat2_> {
  using Type = float2;
};
template <> struct FloatVec<Bfloat4_> {
  using Type = float4;
};
template <> struct FloatVec<Bfloat8_> {
  using Type = Float8_;
};

template <> inline float mul(bfloat16_t a, bfloat16_t b) {
  return (float)a * (float)b;
}
template <> inline bfloat16_t mul(bfloat16_t a, bfloat16_t b) { return a * b; }

template <> inline float2 mul(Bfloat2_ a, Bfloat2_ b) {
  float2 a_f((float)a.x, (float)a.y);
  float2 b_f((float)b.x, (float)b.y);
  return a_f * b_f;
}
template <> inline Bfloat2_ mul(Bfloat2_ a, Bfloat2_ b) {
  Bfloat2_ c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <> inline float4 mul(Bfloat4_ a, Bfloat4_ b) {
  float2 x = mul<float2, Bfloat2_, Bfloat2_>(a.x, b.x);
  float2 y = mul<float2, Bfloat2_, Bfloat2_>(a.y, b.y);
  float4 c;
  c.x = x.x;
  c.y = x.y;
  c.z = y.x;
  c.w = y.y;
  return c;
}
template <> inline Bfloat4_ mul(Bfloat4_ a, Bfloat4_ b) {
  Bfloat4_ c;
  c.x = mul<Bfloat2_, Bfloat2_, Bfloat2_>(a.x, b.x);
  c.y = mul<Bfloat2_, Bfloat2_, Bfloat2_>(a.y, b.y);
  return c;
}

template <> inline Float8_ mul(Bfloat8_ a, Bfloat8_ b) {
  Float8_ c;
  c.x = mul<float4, Bfloat4_, Bfloat4_>(a.x, b.x);
  c.y = mul<float4, Bfloat4_, Bfloat4_>(a.y, b.y);
  return c;
}
template <> inline Bfloat8_ mul(Bfloat8_ a, Bfloat8_ b) {
  Bfloat8_ c;
  c.x = mul<Bfloat4_, Bfloat4_, Bfloat4_>(a.x, b.x);
  c.y = mul<Bfloat4_, Bfloat4_, Bfloat4_>(a.y, b.y);
  return c;
}

template <> inline float sum(bfloat16_t a) { return (float)a; }

template <> inline float sum(Bfloat2_ a) { return (float)a.x + (float)a.y; }

template <> inline float sum(Bfloat4_ a) { return sum(a.x) + sum(a.y); }

template <> inline float sum(Bfloat8_ a) { return sum(a.x) + sum(a.y); }

inline float fma(bfloat16_t a, bfloat16_t b, float c) {
  return (float)a * (float)b + c;
}
inline bfloat16_t fma(bfloat16_t a, bfloat16_t b, bfloat16_t c) {
  return a * b + c;
}

inline float2 fma(Bfloat2_ a, Bfloat2_ b, float2 c) {
  float2 a_f((float)a.x, (float)a.y);
  float2 b_f((float)b.x, (float)b.y);
  return a_f * b_f + c;
}
inline Bfloat2_ fma(Bfloat2_ a, Bfloat2_ b, Bfloat2_ c) {
  Bfloat2_ res;
  res.x = a.x * b.x + c.x;
  res.y = a.y * b.y + c.y;
  return res;
}

inline float4 fma(Bfloat4_ a, Bfloat4_ b, float4 c) {
  float4 res;
  res.x = fma(a.x.x, b.x.x, c.x);
  res.y = fma(a.x.y, b.x.y, c.y);
  res.z = fma(a.y.x, b.y.x, c.z);
  res.w = fma(a.y.y, b.y.y, c.w);
  return res;
}
inline Bfloat4_ fma(Bfloat4_ a, Bfloat4_ b, Bfloat4_ c) {
  Bfloat4_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline Float8_ fma(Bfloat8_ a, Bfloat8_ b, Float8_ c) {
  float4 x = fma(a.x, b.x, c.x);
  float4 y = fma(a.y, b.y, c.y);
  Float8_ res;
  res.x = x;
  res.y = y;
  return res;
}
inline Bfloat8_ fma(Bfloat8_ a, Bfloat8_ b, Bfloat8_ c) {
  Bfloat8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread bfloat16_t &dst, float src) {
  dst = static_cast<bfloat16_t>(src);
}
inline void from_float(thread Bfloat2_ &dst, float2 src) {
  dst.x = static_cast<bfloat16_t>(src.x);
  dst.y = static_cast<bfloat16_t>(src.y);
}
inline void from_float(thread Bfloat4_ &dst, float4 src) {
  dst.x.x = static_cast<bfloat16_t>(src.x);
  dst.x.y = static_cast<bfloat16_t>(src.y);
  dst.y.x = static_cast<bfloat16_t>(src.z);
  dst.y.y = static_cast<bfloat16_t>(src.w);
}
inline void from_float(thread Bfloat8_ &dst, Float8_ src) {
  Bfloat4_ x;
  Bfloat4_ y;
  from_float(x, src.x);
  from_float(y, src.y);
  dst.x = x;
  dst.y = y;
}

// #endif

// ========================================== FP8 (uchar) vector data types.

// 8‑lane uchar vector – Metal only provides up to uchar4, so build our own.
struct Uchar8_ {
  uchar4 x;
  uchar4 y;
};

// Vec specialisations so Vec<uchar, N>::Type resolves correctly.
template <> struct Vec<uchar, 1> {
  using Type = uchar;
};
template <> struct Vec<uchar, 2> {
  using Type = uchar2;
};
template <> struct Vec<uchar, 4> {
  using Type = uchar4;
};
template <> struct Vec<uchar, 8> {
  using Type = Uchar8_;
};

// FP16 vector data types.
struct Half8_ {
  half4 x;
  half4 y;
};

template <> struct Vec<half, 1> {
  using Type = half;
};
template <> struct Vec<half, 2> {
  using Type = half2;
};
template <> struct Vec<half, 4> {
  using Type = half4;
};
template <> struct Vec<half, 8> {
  using Type = Half8_;
};

template <> struct FloatVec<half> {
  using Type = float;
};
template <> struct FloatVec<half2> {
  using Type = float2;
};
template <> struct FloatVec<half4> {
  using Type = float4;
};
template <> struct FloatVec<Half8_> {
  using Type = Float8_;
};

template <> inline float mul(half a, half b) { return (float)a * (float)b; }
template <> inline half mul(half a, half b) { return a * b; }

template <> inline float2 mul(half2 a, half2 b) {
  return (float2)a * (float2)b;
}
template <> inline half2 mul(half2 a, half2 b) { return a * b; }

template <> inline float4 mul(half4 a, half4 b) {
  return (float4)a * (float4)b;
}
template <> inline half4 mul(half4 a, half4 b) { return a * b; }

template <> inline Float8_ mul(Half8_ a, Half8_ b) {
  float4 x = mul<float4, half4, half4>(a.x, b.x);
  float4 y = mul<float4, half4, half4>(a.y, b.y);
  Float8_ c;
  c.x = x;
  c.y = y;
  return c;
}
template <> inline Half8_ mul(Half8_ a, Half8_ b) {
  Half8_ c;
  c.x = mul<half4, half4, half4>(a.x, b.x);
  c.y = mul<half4, half4, half4>(a.y, b.y);
  return c;
}

template <> inline float sum(half a) { return (float)a; }

template <> inline float sum(half2 a) { return (float)a.x + (float)a.y; }

template <> inline float sum(half4 a) { return a.x + a.y + a.z + a.w; }

template <> inline float sum(Half8_ a) { return sum(a.x) + sum(a.y); }

inline float fma(half a, half b, float c) { return (float)a * (float)b + c; }

inline float2 fma(half2 a, half2 b, float2 c) {
  return (float2)a * (float2)b + c;
}

inline float4 fma(half4 a, half4 b, float4 c) {
  return (float4)a * (float4)b + c;
}

inline Float8_ fma(Half8_ a, Half8_ b, Float8_ c) {
  float4 x = fma(a.x, b.x, c.x);
  float4 y = fma(a.y, b.y, c.y);
  Float8_ res;
  res.x = x;
  res.y = y;
  return res;
}
inline Half8_ fma(Half8_ a, Half8_ b, Half8_ c) {
  Half8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread half &dst, float src) {
  dst = static_cast<half>(src);
}
inline void from_float(thread half2 &dst, float2 src) {
  dst.x = static_cast<half>(src.x);
  dst.y = static_cast<half>(src.y);
}
inline void from_float(thread half4 &dst, float4 src) {
  dst.x = static_cast<half>(src.x);
  dst.y = static_cast<half>(src.y);
  dst.z = static_cast<half>(src.z);
  dst.w = static_cast<half>(src.w);
}
inline void from_float(thread Half8_ &dst, Float8_ src) {
  half4 x;
  half4 y;
  from_float(x, src.x);
  from_float(y, src.y);
  dst.x = x;
  dst.y = y;
}

// General case: not uchar
template <typename T> inline constexpr bool is_uchar() { return false; }

// Specialization: T is uchar
template <> inline constexpr bool is_uchar<uchar>() { return true; }

// Generic fallback – will fail to compile if a required specialisation is
// missing.
template <typename Vec, typename Quant_vec>
inline Vec fp8_convert(const thread Quant_vec &, float scale) {
  static_assert(sizeof(Vec) == 0, "Missing fp8_convert specialisation");
}

// ========================================== FP8 -> float/half/bfloat
inline float __dequant_single(uchar v, float scale) {
  return fp8_e4m3_to_float(v) * scale;
}

// ---- 1‑lane ----
template <>
inline float fp8_convert<float, uchar>(const thread uchar &in, float scale) {
  return __dequant_single(in, scale);
}
template <>
inline half fp8_convert<half, uchar>(const thread uchar &in, float scale) {
  return half(__dequant_single(in, scale));
}
template <>
inline bfloat16_t fp8_convert<bfloat16_t, uchar>(const thread uchar &in,
                                                 float scale) {
  return bfloat16_t(__dequant_single(in, scale));
}

// ---- 2‑lane ----
template <>
inline float2 fp8_convert<float2, uchar2>(const thread uchar2 &in,
                                          float scale) {
  return float2(__dequant_single(in.x, scale), __dequant_single(in.y, scale));
}
template <>
inline half2 fp8_convert<half2, uchar2>(const thread uchar2 &in, float scale) {
  half2 out;
  out.x = half(__dequant_single(in.x, scale));
  out.y = half(__dequant_single(in.y, scale));
  return out;
}
template <>
inline Bfloat2_ fp8_convert<Bfloat2_, uchar2>(const thread uchar2 &in,
                                              float scale) {
  Bfloat2_ out;
  out.x = bfloat16_t(__dequant_single(in.x, scale));
  out.y = bfloat16_t(__dequant_single(in.y, scale));
  return out;
}

// ---- 4‑lane ----
template <>
inline float4 fp8_convert<float4, uchar4>(const thread uchar4 &in,
                                          float scale) {
  return float4(__dequant_single(in.x, scale), __dequant_single(in.y, scale),
                __dequant_single(in.z, scale), __dequant_single(in.w, scale));
}
template <>
inline half4 fp8_convert<half4, uchar4>(const thread uchar4 &in, float scale) {
  half4 out;
  out.x = half(__dequant_single(in.x, scale));
  out.y = half(__dequant_single(in.y, scale));
  out.z = half(__dequant_single(in.z, scale));
  out.w = half(__dequant_single(in.w, scale));
  return out;
}
template <>
inline Bfloat4_ fp8_convert<Bfloat4_, uchar4>(const thread uchar4 &in,
                                              float scale) {
  Bfloat4_ out;
  out.x.x = bfloat16_t(__dequant_single(in.x, scale));
  out.x.y = bfloat16_t(__dequant_single(in.y, scale));
  out.y.x = bfloat16_t(__dequant_single(in.z, scale));
  out.y.y = bfloat16_t(__dequant_single(in.w, scale));
  return out;
}

// ---- 8‑lane ----
template <>
inline Float8_ fp8_convert<Float8_, Uchar8_>(const thread Uchar8_ &in,
                                             float scale) {
  Float8_ out;
  out.x =
      float4(__dequant_single(in.x.x, scale), __dequant_single(in.x.y, scale),
             __dequant_single(in.x.z, scale), __dequant_single(in.x.w, scale));
  out.y =
      float4(__dequant_single(in.y.x, scale), __dequant_single(in.y.y, scale),
             __dequant_single(in.y.z, scale), __dequant_single(in.y.w, scale));
  return out;
}
template <>
inline Half8_ fp8_convert<Half8_, Uchar8_>(const thread Uchar8_ &in,
                                           float scale) {
  Half8_ out;
  out.x = half4(half(__dequant_single(in.x.x, scale)),
                half(__dequant_single(in.x.y, scale)),
                half(__dequant_single(in.x.z, scale)),
                half(__dequant_single(in.x.w, scale)));
  out.y = half4(half(__dequant_single(in.y.x, scale)),
                half(__dequant_single(in.y.y, scale)),
                half(__dequant_single(in.y.z, scale)),
                half(__dequant_single(in.y.w, scale)));
  return out;
}
template <>
inline Bfloat8_ fp8_convert<Bfloat8_, Uchar8_>(const thread Uchar8_ &in,
                                               float scale) {
  Bfloat8_ out;
  // first 4
  out.x.x.x = bfloat16_t(__dequant_single(in.x.x, scale));
  out.x.x.y = bfloat16_t(__dequant_single(in.x.y, scale));
  out.x.y.x = bfloat16_t(__dequant_single(in.x.z, scale));
  out.x.y.y = bfloat16_t(__dequant_single(in.x.w, scale));
  // second 4
  out.y.x.x = bfloat16_t(__dequant_single(in.y.x, scale));
  out.y.x.y = bfloat16_t(__dequant_single(in.y.y, scale));
  out.y.y.x = bfloat16_t(__dequant_single(in.y.z, scale));
  out.y.y.y = bfloat16_t(__dequant_single(in.y.w, scale));
  return out;
}

// ========================================== Dot product utilities

template <int THREAD_GROUP_SIZE, typename Vec, int N>
inline float qk_dot_(const threadgroup Vec (&q)[N], const thread Vec (&k)[N]) {
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  using A_vec = typename FloatVec<Vec>::Type;
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += simd_shuffle_xor(qk, mask);
  }
  return qk;
}

template <typename T, int THREAD_GROUP_SIZE> struct Qk_dot {
  template <typename Vec, int N>
  static inline float dot(const threadgroup Vec (&q)[N],
                          const thread Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};

// ========================================== Block sum utility

// Utility function for attention softmax.
template <int NUM_WARPS, int NUM_SIMD_LANES>
inline float block_sum(threadgroup float *red_smem, float sum, uint simd_tid,
                       uint simd_lid) {
  // Compute the sum per simdgroup.
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }

  // Simd leaders store the data to shared memory.
  if (simd_lid == 0) {
    red_smem[simd_tid] = sum;
  }

  // Make sure the data is in shared memory.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The warps compute the final sums.
  if (simd_lid < NUM_WARPS) {
    sum = red_smem[simd_lid];
  }

  // Parallel reduction inside the simd group.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }

  // Broadcast to other threads.
  return simd_shuffle(sum, 0);
}

// ========================================== Paged Attention kernel

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))


// ===== to_cache helpers (from reshape_and_cache.metal) =====


template <typename KV_T, typename CACHE_T>
inline CACHE_T to_cache(KV_T v) = delete;

template <> inline uchar to_cache<float, uchar>(float v) {
  return float_to_fp8_e4m3(v);
}

template <> inline uchar to_cache<bfloat16_t, uchar>(bfloat16_t v) {
  return float_to_fp8_e4m3((float)v);
}

template <> inline uchar to_cache<half, uchar>(half v) {
  return float_to_fp8_e4m3((float)v);
}

template <> inline float to_cache<float, float>(float v) { return v; }

template <> inline bfloat16_t to_cache<bfloat16_t, bfloat16_t>(bfloat16_t v) {
  return v;
}

template <> inline half to_cache<half, half>(half v) { return v; }

constant bool use_fp8_scales [[function_constant(10)]];


// ===== from_cache helpers (from gather_kv_cache.metal) =====


// Convert from cache type to output type, with optional FP8 dequantization.
template <typename CACHE_T, typename OUT_T>
inline OUT_T from_cache(CACHE_T v) = delete;

// Identity conversions (cache_t == out_t)
template <> inline float from_cache<float, float>(float v) { return v; }
template <> inline bfloat16_t from_cache<bfloat16_t, bfloat16_t>(bfloat16_t v) {
  return v;
}
template <> inline half from_cache<half, half>(half v) { return v; }

// FP8 E4M3 -> output type conversions
template <> inline float from_cache<uchar, float>(uchar v) {
  return fp8_e4m3_to_float(v);
}
template <> inline half from_cache<uchar, half>(uchar v) {
  return (half)fp8_e4m3_to_float(v);
}
template <> inline bfloat16_t from_cache<uchar, bfloat16_t>(uchar v) {
  return (bfloat16_t)fp8_e4m3_to_float(v);
}


/// Gather K and V from paged KV cache into contiguous output tensors.
///
/// One threadgroup per output token. Threads cooperatively copy
/// kv_heads * head_size elements for both K and V.
///
/// Uses binary search on cu_seq_lens to find batch_id.
///
/// K cache layout: [num_blocks, kv_heads, head_size/x, block_size, x]
/// V cache layout: [num_blocks, kv_heads, head_size, block_size]
/// K/V output:     [num_tokens, kv_heads, head_size]


// ===== kv_scale_update helpers (from kv_scale_update.metal) =====


#define DIV_CONST 240.0f


// ===== Utility macros =====
"""
