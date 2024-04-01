#include "pbrt/euclidean_space/squared_matrix.cuh"

template <>
PBRT_CPU_GPU double SquareMatrix<3>::determinant() const {
    const auto m = this->val;

    double minor12 = difference_of_products(m[1][1], m[2][2], m[1][2], m[2][1]);
    double minor02 = difference_of_products(m[1][0], m[2][2], m[1][2], m[2][0]);
    double minor01 = difference_of_products(m[1][0], m[2][1], m[1][1], m[2][0]);

    return std::fma(m[0][2], minor01, difference_of_products(m[0][0], minor12, m[0][1], minor02));
}

template <>
PBRT_CPU_GPU double SquareMatrix<4>::determinant() const {
    const auto m = this->val;

    double s0 = difference_of_products(m[0][0], m[1][1], m[1][0], m[0][1]);
    double s1 = difference_of_products(m[0][0], m[1][2], m[1][0], m[0][2]);
    double s2 = difference_of_products(m[0][0], m[1][3], m[1][0], m[0][3]);

    double s3 = difference_of_products(m[0][1], m[1][2], m[1][1], m[0][2]);
    double s4 = difference_of_products(m[0][1], m[1][3], m[1][1], m[0][3]);
    double s5 = difference_of_products(m[0][2], m[1][3], m[1][2], m[0][3]);

    double c0 = difference_of_products(m[2][0], m[3][1], m[3][0], m[2][1]);
    double c1 = difference_of_products(m[2][0], m[3][2], m[3][0], m[2][2]);
    double c2 = difference_of_products(m[2][0], m[3][3], m[3][0], m[2][3]);

    double c3 = difference_of_products(m[2][1], m[3][2], m[3][1], m[2][2]);
    double c4 = difference_of_products(m[2][1], m[3][3], m[3][1], m[2][3]);
    double c5 = difference_of_products(m[2][2], m[3][3], m[3][2], m[2][3]);

    return difference_of_products(s0, c5, s1, c4) + difference_of_products(s2, c3, -s3, c2) +
           difference_of_products(s5, c0, s4, c1);
}

template <uint N>
PBRT_CPU_GPU double SquareMatrix<N>::determinant() const {
    printf("determinant() not implemented for SquareMatrix<%d>\n", N);

#if defined(__CUDA_ARCH__)
    asm("trap;");
#else
    throw std::runtime_error("SquareMatrix<N>::determinant() not implemented");
#endif
}

template <>
PBRT_CPU_GPU SquareMatrix<3> SquareMatrix<3>::inverse() const {
    const double det = determinant();
    if (det == 0.0) {
        printf("can't inverse a singular matrix\n");
#if defined(__CUDA_ARCH__)
        asm("trap;");
#else
        throw std::runtime_error("singular matrix");
#endif
    }

    const double inv_det = 1.0 / det;

    const auto m = this->val;

    double r[3][3];
    r[0][0] = inv_det * difference_of_products(m[1][1], m[2][2], m[1][2], m[2][1]);
    r[1][0] = inv_det * difference_of_products(m[1][2], m[2][0], m[1][0], m[2][2]);
    r[2][0] = inv_det * difference_of_products(m[1][0], m[2][1], m[1][1], m[2][0]);
    r[0][1] = inv_det * difference_of_products(m[0][2], m[2][1], m[0][1], m[2][2]);
    r[1][1] = inv_det * difference_of_products(m[0][0], m[2][2], m[0][2], m[2][0]);
    r[2][1] = inv_det * difference_of_products(m[0][1], m[2][0], m[0][0], m[2][1]);
    r[0][2] = inv_det * difference_of_products(m[0][1], m[1][2], m[0][2], m[1][1]);
    r[1][2] = inv_det * difference_of_products(m[0][2], m[1][0], m[0][0], m[1][2]);
    r[2][2] = inv_det * difference_of_products(m[0][0], m[1][1], m[0][1], m[1][0]);

    return SquareMatrix(r);
}

template <>
PBRT_CPU_GPU SquareMatrix<4> SquareMatrix<4>::inverse() const {
    /*
Via: https://github.com/google/ion/blob/master/ion/math/matrixutils.cc,
(c) Google, Apache license.

    For 4x4 do not compute the adjugate as the transpose of the cofactor
        matrix, because this results in extra work. Several calculations can be
                shared across the sub-determinants.
                //
                This approach is explained in David Eberly's Geometric Tools book,
                excerpted here:
        http://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf
                   */

    const auto m = this->val;

    double s0 = difference_of_products(m[0][0], m[1][1], m[1][0], m[0][1]);
    double s1 = difference_of_products(m[0][0], m[1][2], m[1][0], m[0][2]);
    double s2 = difference_of_products(m[0][0], m[1][3], m[1][0], m[0][3]);

    double s3 = difference_of_products(m[0][1], m[1][2], m[1][1], m[0][2]);
    double s4 = difference_of_products(m[0][1], m[1][3], m[1][1], m[0][3]);
    double s5 = difference_of_products(m[0][2], m[1][3], m[1][2], m[0][3]);

    double c0 = difference_of_products(m[2][0], m[3][1], m[3][0], m[2][1]);
    double c1 = difference_of_products(m[2][0], m[3][2], m[3][0], m[2][2]);
    double c2 = difference_of_products(m[2][0], m[3][3], m[3][0], m[2][3]);

    double c3 = difference_of_products(m[2][1], m[3][2], m[3][1], m[2][2]);
    double c4 = difference_of_products(m[2][1], m[3][3], m[3][1], m[2][3]);
    double c5 = difference_of_products(m[2][2], m[3][3], m[3][2], m[2][3]);

    double determinant = inner_product(s0, c5, -s1, c4, s2, c3, s3, c2, s5, c0, -s4, c1);
    if (determinant == 0) {
        printf("can't inverse a singular matrix\n");
#if defined(__CUDA_ARCH__)
        asm("trap;");
#else
        throw std::runtime_error("singular matrix");
#endif
    }
    double s = 1.0 / determinant;

    double inv[4][4] = {{
                            s * inner_product(m[1][1], c5, m[1][3], c3, -m[1][2], c4),
                            s * inner_product(-m[0][1], c5, m[0][2], c4, -m[0][3], c3),
                            s * inner_product(m[3][1], s5, m[3][3], s3, -m[3][2], s4),
                            s * inner_product(-m[2][1], s5, m[2][2], s4, -m[2][3], s3),
                        },
                        {
                            s * inner_product(-m[1][0], c5, m[1][2], c2, -m[1][3], c1),
                            s * inner_product(m[0][0], c5, m[0][3], c1, -m[0][2], c2),
                            s * inner_product(-m[3][0], s5, m[3][2], s2, -m[3][3], s1),
                            s * inner_product(m[2][0], s5, m[2][3], s1, -m[2][2], s2),
                        },
                        {
                            s * inner_product(m[1][0], c4, m[1][3], c0, -m[1][1], c2),
                            s * inner_product(-m[0][0], c4, m[0][1], c2, -m[0][3], c0),
                            s * inner_product(m[3][0], s4, m[3][3], s0, -m[3][1], s2),
                            s * inner_product(-m[2][0], s4, m[2][1], s2, -m[2][3], s0),
                        },
                        {
                            s * inner_product(-m[1][0], c3, m[1][1], c1, -m[1][2], c0),
                            s * inner_product(m[0][0], c3, m[0][2], c0, -m[0][1], c1),
                            s * inner_product(-m[3][0], s3, m[3][1], s1, -m[3][2], s0),
                            s * inner_product(m[2][0], s3, m[2][2], s0, -m[2][1], s1),
                        }};

    return SquareMatrix(inv);
}

template <uint N>
PBRT_CPU_GPU SquareMatrix<N> SquareMatrix<N>::inverse() const {
    printf("inverse() not implemented for SquareMatrix<%u>\n", N);

#if defined(__CUDA_ARCH__)
    asm("trap;");
#else
    throw std::runtime_error("SquareMatrix<N>::inverse() not implemented");
#endif
}
