#pragma once

#include "pbrt/util/rounding_arithmetic.h"
#include "accurate_arithmetic.h"

class Interval {
  public:
    double low;
    double high;

    PBRT_CPU_GPU Interval() : low(NAN), high(NAN) {}

    PBRT_CPU_GPU
    explicit Interval(double v) : low(v), high(v) {}
    PBRT_CPU_GPU constexpr Interval(double low, double high)
        : low(std::min(low, high)), high(std::max(low, high)) {}

    PBRT_CPU_GPU
    static Interval from_value_and_error(double v, double err) {
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

    PBRT_CPU_GPU double midpoint() const {
        return (low + high) / 2;
    }

    PBRT_CPU_GPU
    explicit operator double() const {
        return midpoint();
    }

    PBRT_CPU_GPU double width() const {
        return high - low;
    }

    PBRT_CPU_GPU
    bool exactly(double v) const {
        return low == v && high == v;
    }

    PBRT_CPU_GPU
    bool operator==(double v) const {
        return exactly(v);
    }

    PBRT_CPU_GPU Interval operator+(double f) const {
        return (*this) + Interval(f);
    }

    PBRT_CPU_GPU
    Interval operator+(const Interval &i) const {
        return {add_round_down(low, i.low), add_round_up(high, i.high)};
    }

    PBRT_CPU_GPU
    void operator+=(double f) {
        (*this) = (*this) + f;
    }

    PBRT_CPU_GPU Interval operator*(double f) const {
        return f > 0.0 ? Interval(mul_round_down(f, low), mul_round_up(f, high))
                       : Interval(mul_round_down(f, high), mul_round_up(f, low));
    }

    PBRT_CPU_GPU Interval operator/(const Interval &i) const;
};

// Interval Inline Functions
PBRT_CPU_GPU inline bool in_range(double v, Interval i) {
    return v >= i.low && v <= i.high;
}
PBRT_CPU_GPU inline bool in_range(Interval a, Interval b) {
    return a.low <= b.high && a.high >= b.low;
}

PBRT_GPU inline Interval sqr(Interval i) {
    double alow = std::abs(i.low);
    double ahigh = std::abs(i.high);
    if (alow > ahigh) {
        std::swap(alow, ahigh);
    }

    if (in_range(0, i)) {
        return Interval(0, mul_round_up(ahigh, ahigh));
    }

    return Interval(mul_round_down(alow, alow), mul_round_up(ahigh, ahigh));
}

PBRT_CPU_GPU inline Interval operator*(double f, const Interval &i) {
    return i * f;
}

PBRT_CPU_GPU inline Interval operator+(double f, const Interval &i) {
    return i + f;
}

PBRT_CPU_GPU Interval Interval::operator/(const Interval &i) const {
    if (in_range(0, i)) {
        // The interval we're dividing by straddles zero, so just
        // return an interval of everything.

        return Interval(-Infinity, Infinity);
    }

    double lowQuot[4] = {div_round_down(low, i.low), div_round_down(high, i.low),
                         div_round_down(low, i.high), div_round_down(high, i.high)};
    double highQuot[4] = {div_round_up(low, i.low), div_round_up(high, i.low),
                          div_round_up(low, i.high), div_round_up(high, i.high)};
    return {std::min({lowQuot[0], lowQuot[1], lowQuot[2], lowQuot[3]}),
            std::max({highQuot[0], highQuot[1], highQuot[2], highQuot[3]})};
}
