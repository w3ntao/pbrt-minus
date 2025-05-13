#pragma once

#include <pbrt/util/math.h>
#include <pbrt/util/bit_arithmetic.h>
#include <pbrt/util/util.h>
#include <algorithm>

class Interval {
  public:
    Real low;
    Real high;

    PBRT_CPU_GPU
    Interval() : low(NAN), high(NAN) {}

    PBRT_CPU_GPU
    explicit Interval(Real v) : low(v), high(v) {}

    PBRT_CPU_GPU
    Interval(Real low, Real high) : low(std::min(low, high)), high(std::max(low, high)) {}

    PBRT_CPU_GPU
    static Interval from_value_and_error(Real v, Real err) {
        Interval i;
        if (err == 0) {
            i.low = v;
            i.high = v;
        } else {
            i.low = sub_round_down(v, err);
            i.high = add_round_up(v, err);
        }
        return i;
    }

    PBRT_CPU_GPU
    Real midpoint() const {
        return (low + high) / 2;
    }

    PBRT_CPU_GPU
    explicit operator Real() const {
        return midpoint();
    }

    PBRT_CPU_GPU
    Real width() const {
        return high - low;
    }

    PBRT_CPU_GPU
    bool has_nan() const {
        return isnan(low) || isnan(high) || isinf(low) || isinf(high);
    }

    PBRT_CPU_GPU
    bool exactly(Real v) const {
        return low == v && high == v;
    }

    PBRT_CPU_GPU
    bool operator==(Real v) const {
        return exactly(v);
    }

    PBRT_CPU_GPU bool operator==(const Interval &i) const {
        return low == i.low && high == i.high;
    }

    PBRT_CPU_GPU
    bool cover(Real v) const {
        return v >= low && v <= high;
    }

    PBRT_CPU_GPU
    Interval operator+(const Interval &i) const {
        return {add_round_down(low, i.low), add_round_up(high, i.high)};
    }

    PBRT_CPU_GPU Interval operator+(Real f) const {
        return (*this) + Interval(f);
    }

    PBRT_CPU_GPU
    void operator+=(Real f) {
        (*this) = (*this) + f;
    }

    PBRT_CPU_GPU
    Interval operator-(const Interval &i) const {
        return {sub_round_down(low, i.high), sub_round_up(high, i.low)};
    }

    PBRT_CPU_GPU
    Interval operator*(Real f) const {
        return f > 0.0 ? Interval(mul_round_down(f, low), mul_round_up(f, high))
                       : Interval(mul_round_down(f, high), mul_round_up(f, low));
    }

    PBRT_CPU_GPU
    Interval operator*(Interval i) const {
        Real lp[4] = {mul_round_down(low, i.low), mul_round_down(high, i.low),
                           mul_round_down(low, i.high), mul_round_down(high, i.high)};
        Real hp[4] = {mul_round_up(low, i.low), mul_round_up(high, i.low),
                           mul_round_up(low, i.high), mul_round_up(high, i.high)};
        return {std::min({lp[0], lp[1], lp[2], lp[3]}), std::max({hp[0], hp[1], hp[2], hp[3]})};
    }

    PBRT_CPU_GPU
    inline Interval operator/(const Interval &i) const {
        if (i.cover(0)) {
            // The interval we're dividing by straddles zero, so just
            // return an interval of everything.

            return Interval(-Infinity, Infinity);
        }

        Real lowQuot[4] = {div_round_down(low, i.low), div_round_down(high, i.low),
                                div_round_down(low, i.high), div_round_down(high, i.high)};
        Real highQuot[4] = {div_round_up(low, i.low), div_round_up(high, i.low),
                                 div_round_up(low, i.high), div_round_up(high, i.high)};
        return {std::min({lowQuot[0], lowQuot[1], lowQuot[2], lowQuot[3]}),
                std::max({highQuot[0], highQuot[1], highQuot[2], highQuot[3]})};
    }

    PBRT_CPU_GPU
    inline Interval operator/(Real f) const {
        if (f == 0) {
            return Interval(-Infinity, Infinity);
        }

        if (f > 0) {
            return Interval(div_round_down(low, f), div_round_up(high, f));
        } else {
            return Interval(div_round_down(high, f), div_round_up(low, f));
        }
    }

    PBRT_CPU_GPU Interval sqrt() const {
        return {sqrt_round_down(low), sqrt_round_up(high)};
    }
};

PBRT_CPU_GPU
inline Interval sqr(const Interval &i) {
    Real abs_low = std::abs(i.low);
    Real abs_high = std::abs(i.high);
    if (abs_low > abs_high) {
        pbrt::swap(abs_low, abs_high);
    }

    if (i.cover(0)) {
        return Interval(0, mul_round_up(abs_high, abs_high));
    }

    return Interval(mul_round_down(abs_low, abs_low), mul_round_up(abs_high, abs_high));
}

PBRT_CPU_GPU
inline Interval operator*(Real f, const Interval &i) {
    return i * f;
}

PBRT_CPU_GPU
inline Interval operator+(Real f, const Interval &i) {
    return i + f;
}
