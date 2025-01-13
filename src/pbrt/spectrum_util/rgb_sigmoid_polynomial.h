#pragma once

#include <pbrt/spectrum_util/spectrum_constants_cie.h>
#include <pbrt/util/basic_math.h>

class RGBSigmoidPolynomial {
  public:
    PBRT_CPU_GPU
    RGBSigmoidPolynomial() : c0(NAN), c1(NAN), c2(NAN) {}

    PBRT_CPU_GPU
    RGBSigmoidPolynomial(FloatType c0, FloatType c1, FloatType c2) : c0(c0), c1(c1), c2(c2) {}

    PBRT_CPU_GPU
    FloatType operator()(FloatType lambda) const {
        return s(evaluate_polynomial(lambda, c2, c1, c0));
    }

    PBRT_CPU_GPU
    FloatType max_value() const {
        FloatType result = std::max((*this)(LAMBDA_MIN), (*this)(LAMBDA_MAX));
        FloatType lambda = -c1 / (2 * c0);

        if (lambda >= LAMBDA_MIN && lambda <= LAMBDA_MAX) {
            result = std::max(result, (*this)(lambda));
        }

        return result;
    }

  private:
    PBRT_CPU_GPU
    static FloatType s(FloatType x) {
        if (is_inf(x)) {
            return x > 0 ? 1 : 0;
        }

        return .5f + x / (2 * std::sqrt(1 + sqr(x)));
    }

    FloatType c0, c1, c2;
};
