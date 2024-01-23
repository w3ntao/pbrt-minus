#pragma once

#include "pbrt/util/rounding_arithmetic.h"

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
    static Interval FromValueAndError(double v, double err) {
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

    if (in_range(0, i))
        return Interval(0, mul_round_up(ahigh, ahigh));
    return Interval(mul_round_down(alow, alow), mul_round_up(ahigh, ahigh));
}
