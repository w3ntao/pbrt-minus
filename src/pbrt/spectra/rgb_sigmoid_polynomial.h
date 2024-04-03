#pragma once

#include "pbrt/util/utility_math.h"
#include "pbrt/spectra/constants.h"

class RGBSigmoidPolynomial {
  public:
    PBRT_CPU_GPU
    RGBSigmoidPolynomial() : c0(NAN), c1(NAN), c2(NAN) {}

    PBRT_CPU_GPU
    RGBSigmoidPolynomial(double c0, double c1, double c2) : c0(c0), c1(c1), c2(c2) {}

    PBRT_CPU_GPU
    double operator()(double lambda) const {
        return s(evaluate_polynomial(lambda, c2, c1, c0));
    }

    PBRT_CPU_GPU
    double max_value() const {
        double result = std::max((*this)(LAMBDA_MIN), (*this)(LAMBDA_MAX));
        double lambda = -c1 / (2 * c0);

        if (lambda >= LAMBDA_MIN && lambda <= LAMBDA_MAX) {
            result = std::max(result, (*this)(lambda));
        }

        return result;
    }

  private:
    PBRT_CPU_GPU
    static double s(double x) {
        if (is_inf(x)) {
            return x > 0 ? 1 : 0;
        }

        return .5f + x / (2 * std::sqrt(1 + sqr(x)));
    }

    double c0, c1, c2;
};
