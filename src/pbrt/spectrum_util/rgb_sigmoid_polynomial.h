#pragma once

#include <pbrt/spectrum_util/spectrum_constants_cie.h>
#include <pbrt/util/math.h>

class RGBSigmoidPolynomial {
  public:
    PBRT_CPU_GPU
    RGBSigmoidPolynomial() {}

    PBRT_CPU_GPU
    RGBSigmoidPolynomial(Real _c0, Real _c1, Real _c2) : c0(_c0), c1(_c1), c2(_c2) {}

    PBRT_CPU_GPU
    Real operator()(Real lambda) const {
        return s(evaluate_polynomial(lambda, c2, c1, c0));
    }

    PBRT_CPU_GPU
    Real max_value() const {
        Real result = std::max((*this)(LAMBDA_MIN), (*this)(LAMBDA_MAX));
        Real lambda = -c1 / (2 * c0);

        if (lambda >= LAMBDA_MIN && lambda <= LAMBDA_MAX) {
            result = std::max(result, (*this)(lambda));
        }

        return result;
    }

  private:
    PBRT_CPU_GPU
    static Real s(Real x) {
        if (is_inf(x)) {
            return x > 0 ? 1 : 0;
        }

        return .5f + x / (2 * std::sqrt(1 + sqr(x)));
    }

    Real c0 = NAN;
    Real c1 = NAN;
    Real c2 = NAN;
};
