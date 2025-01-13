#pragma once

#include <pbrt/gpu/macro.h>
#include <cuda_fp16.h>

static const int HalfExponentMask = 0b0111110000000000;
static const int HalfSignificandMask = 0b1111111111;
static const int HalfNegativeZero = 0b1000000000000000;
static const int HalfPositiveZero = 0;
// Exponent all 1s, significand zero
static const int HalfNegativeInfinity = 0b1111110000000000;
static const int HalfPositiveInfinity = 0b0111110000000000;

union FP16 {
    uint16_t u;
    struct {
        unsigned int Mantissa : 10;
        unsigned int Exponent : 5;
        unsigned int Sign : 1;
    };
};

union FP32 {
    uint32_t u;
    float f;
    struct {
        unsigned int Mantissa : 23;
        unsigned int Exponent : 8;
        unsigned int Sign : 1;
    };
};

class Half {
  public:
    Half() = default;
    Half(const Half &) = default;
    Half &operator=(const Half &) = default;

    PBRT_CPU_GPU
    static Half FromBits(uint16_t v) {
        return Half(v);
    }

    PBRT_CPU_GPU
    explicit Half(float ff) {
#if defined(__CUDA_ARCH__)
        h = __half_as_ushort(__float2half(ff));
#else
        // Rounding ties to nearest even instead of towards +inf
        FP32 f;
        f.f = ff;
        FP32 f32infty = {255 << 23};
        FP32 f16max = {(127 + 16) << 23};
        FP32 denorm_magic = {((127 - 15) + (23 - 10) + 1) << 23};
        unsigned int sign_mask = 0x80000000u;
        FP16 o = {0};

        unsigned int sign = f.u & sign_mask;
        f.u ^= sign;

        // NOTE all the integer compares in this function can be safely
        // compiled into signed compares since all operands are below
        // 0x80000000. Important if you want fast straight SSE2 code
        // (since there's no unsigned PCMPGTD).

        if (f.u >= f16max.u) { // result is Inf or NaN (all exponent bits set)
            o.u = (f.u > f32infty.u) ? 0x7e00 : 0x7c00;
        }                            // NaN->qNaN and Inf->Inf
        else {                       // (De)normalized number or zero
            if (f.u < (113 << 23)) { // resulting FP16 is subnormal or zero
                // use a magic value to align our 10 mantissa bits at the bottom
                // of the float. as long as FP addition is round-to-nearest-even
                // this just works.
                f.f += denorm_magic.f;

                // and one integer subtract of the bias later, we have our final
                // float!
                o.u = f.u - denorm_magic.u;
            } else {
                unsigned int mant_odd = (f.u >> 13) & 1; // resulting mantissa is odd

                // update exponent, rounding bias part 1
                f.u += (uint32_t(15 - 127) << 23) + 0xfff;
                // rounding bias part 2
                f.u += mant_odd;
                // take the bits!
                o.u = f.u >> 13;
            }
        }

        o.u |= sign >> 16;
        h = o.u;
#endif
    }
    PBRT_CPU_GPU
    explicit Half(double d) : Half(float(d)) {}

    PBRT_CPU_GPU
    explicit operator float() const {
#if defined(__CUDA_ARCH__)
        return __half2float(__ushort_as_half(h));
#else
        FP16 h;
        h.u = this->h;
        static const FP32 magic = {113 << 23};
        static const unsigned int shifted_exp = 0x7c00 << 13; // exponent mask after shift
        FP32 o;

        o.u = (h.u & 0x7fff) << 13;           // exponent/mantissa bits
        unsigned int exp = shifted_exp & o.u; // just the exponent
        o.u += (127 - 15) << 23;              // exponent adjust

        // handle exponent special cases
        if (exp == shifted_exp) {
            // Inf/NaN?
            o.u += (128 - 16) << 23; // extra exp adjust
        } else if (exp == 0) {       // Zero/Denormal?
            o.u += 1 << 23;          // extra exp adjust
            o.f -= magic.f;          // renormalize
        }

        o.u |= (h.u & 0x8000) << 16; // sign bit
        return o.f;
#endif
    }
    PBRT_CPU_GPU
    explicit operator double() const {
        return (float)(*this);
    }

    PBRT_CPU_GPU
    bool operator==(const Half &v) const {
#if defined(__CUDA_ARCH__)
        return __ushort_as_half(h) == __ushort_as_half(v.h);
#else
        if (Bits() == v.Bits()) {
            return true;
        }

        return ((Bits() == HalfNegativeZero && v.Bits() == HalfPositiveZero) ||
                (Bits() == HalfPositiveZero && v.Bits() == HalfNegativeZero));
#endif
    }
    PBRT_CPU_GPU
    bool operator!=(const Half &v) const {
        return !(*this == v);
    }

    PBRT_CPU_GPU
    Half operator-() const {
        return FromBits(h ^ (1 << 15));
    }

    PBRT_CPU_GPU
    uint16_t Bits() const {
        return h;
    }

    PBRT_CPU_GPU
    int Sign() {
        return (h >> 15) ? -1 : 1;
    }

    PBRT_CPU_GPU
    bool IsInf() {
        return h == HalfPositiveInfinity || h == HalfNegativeInfinity;
    }

    PBRT_CPU_GPU
    bool IsNaN() {
        return ((h & HalfExponentMask) == HalfExponentMask && (h & HalfSignificandMask) != 0);
    }

    PBRT_CPU_GPU
    Half NextUp() {
        if (IsInf() && Sign() == 1) {
            return *this;
        }

        Half up = *this;
        if (up.h == HalfNegativeZero) {
            up.h = HalfPositiveZero;
        }

        // Advance _v_ to next higher float
        if (up.Sign() >= 0) {
            ++up.h;
        } else {
            --up.h;
        }

        return up;
    }

    PBRT_CPU_GPU
    Half NextDown() {
        if (IsInf() && Sign() == -1)
            return *this;

        Half down = *this;
        if (down.h == HalfPositiveZero) {
            down.h = HalfNegativeZero;
        }
        if (down.Sign() >= 0) {
            --down.h;
        } else {
            ++down.h;
        }

        return down;
    }

  private:
    PBRT_CPU_GPU
    explicit Half(uint16_t h) : h(h) {}

    uint16_t h;
};
