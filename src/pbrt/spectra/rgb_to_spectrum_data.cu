#include "pbrt/spectra/rgb_to_spectrum_data.h"
#include "pbrt/util/thread_pool.h"

constexpr double RGB2SPEC_EPSILON = 1e-4;

const double xyz_to_srgb[3][3] = {
    {3.240479, -1.537150, -0.498535},
    {-0.969256, 1.875991, 0.041556},
    {0.055648, -0.204043, 1.057311},
};

const double srgb_to_xyz[3][3] = {
    {0.412453, 0.357580, 0.180423},
    {0.212671, 0.715160, 0.072169},
    {0.019334, 0.119193, 0.950227},
};

[[maybe_unused]] const double xyz_to_xyz[3][3] = {
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0},
};

double sigmoid(double x) {
    return 0.5 * x / std::sqrt(1.0 + x * x) + 0.5;
}

double smoothstep(double x) {
    return x * x * (3.0 - 2.0 * x);
}

namespace RGBtoSpectrumData {

struct RGBtoSpectrumBuffer {
    double lambda_tbl[CIE_FINE_SAMPLES];
    double rgb_tbl[3][CIE_FINE_SAMPLES];
    double rgb_to_xyz[3][3];
    double xyz_to_rgb[3][3];
    double xyz_whitepoint[3];
};

double sqr(double x) {
    return x * x;
}

double cie_interp(const double *data, double x) {
    x -= CIE_LAMBDA_MIN;
    x *= (CIE_SAMPLES - 1) / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);
    int offset = (int)x;

    if (offset < 0) {
        offset = 0;
    }

    if (offset > CIE_SAMPLES - 2) {
        offset = CIE_SAMPLES - 2;
    }

    double weight = x - offset;
    return (1.0 - weight) * data[offset] + weight * data[offset + 1];
}

int LUPDecompose(double **A, int N, double Tol, int *P) {
    int i, j, k, imax;
    double maxA, *ptr, absA;

    for (i = 0; i <= N; i++) {
        P[i] = i;
        // Unit permutation matrix, P[N] initialized with N
    }

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        for (k = i; k < N; k++) {
            if ((absA = fabs(A[k][i])) > maxA) {
                maxA = absA;
                imax = k;
            }
        }

        if (maxA < Tol) {
            return 0; // failure, matrix is degenerate
        }

        if (imax != i) {
            // pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            // pivoting rows of A
            ptr = A[i];
            A[i] = A[imax];
            A[imax] = ptr;

            // counting pivots starting from N (for determinant)
            P[N]++;
        }

        for (j = i + 1; j < N; j++) {
            A[j][i] /= A[i][i];

            for (k = i + 1; k < N; k++) {
                A[j][k] -= A[j][i] * A[i][k];
            }
        }
    }

    return 1; // decomposition done
}

/* INPUT: A,P filled in LUPDecompose; b - rhs vector; N - dimension
 * OUTPUT: x - solution vector of A*x=b
 */
void LUPSolve(double **const A, const int *P, const double *b, int N, double *x) {
    for (int i = 0; i < N; i++) {
        x[i] = b[P[i]];

        for (int k = 0; k < i; k++) {
            x[i] -= A[i][k] * x[k];
        }
    }

    for (int i = N - 1; i >= 0; i--) {
        for (int k = i + 1; k < N; k++) {
            x[i] -= A[i][k] * x[k];
        }

        x[i] = x[i] / A[i][i];
    }
}

void cie_lab(double *p, const RGBtoSpectrumBuffer *data) {
    double X = 0.0;
    double Y = 0.0;
    double Z = 0.0;
    double Xw = data->xyz_whitepoint[0];
    double Yw = data->xyz_whitepoint[1];
    double Zw = data->xyz_whitepoint[2];

    for (int j = 0; j < 3; ++j) {
        X += p[j] * data->rgb_to_xyz[0][j];
        Y += p[j] * data->rgb_to_xyz[1][j];
        Z += p[j] * data->rgb_to_xyz[2][j];
    }

    auto f = [](double t) -> double {
        double delta = 6.0 / 29.0;
        if (t > delta * delta * delta) {
            return cbrt(t);
        }

        return t / (delta * delta * 3.0) + (4.0 / 29.0);
    };

    p[0] = 116.0 * f(Y / Yw) - 16.0;
    p[1] = 500.0 * (f(X / Xw) - f(Y / Yw));
    p[2] = 200.0 * (f(Y / Yw) - f(Z / Zw));
}

void init_tables(RGBtoSpectrumBuffer *data, Gamut gamut) {
    memset(data->rgb_tbl, 0, sizeof(data->rgb_tbl));
    memset(data->xyz_whitepoint, 0, sizeof(data->xyz_whitepoint));

    double h = double(CIE_LAMBDA_MAX - CIE_LAMBDA_MIN) / double(CIE_FINE_SAMPLES - 1);

    const double *illuminant = nullptr;

    switch (gamut) {
    case Gamut::sRGB: {
        illuminant = cie_d65;
        memcpy(data->xyz_to_rgb, xyz_to_srgb, sizeof(double) * 9);
        memcpy(data->rgb_to_xyz, srgb_to_xyz, sizeof(double) * 9);
        break;
    }

    default: {
        throw std::runtime_error("init_tables(): invalid/unsupported gamut.");
    }
    }

    for (uint i = 0; i < CIE_FINE_SAMPLES; ++i) {
        double lambda = CIE_LAMBDA_MIN + i * h;

        double xyz[3] = {cie_interp(cie_x, lambda), cie_interp(cie_y, lambda),
                         cie_interp(cie_z, lambda)};
        double I = cie_interp(illuminant, lambda);

        double ratio = 1.0;
        if (i == 0 || i == CIE_FINE_SAMPLES - 1) {
            ratio = 1.0;
        } else if ((i - 1) % 3 == 2) {
            ratio = 2.0;
        } else {
            ratio = 3.0;
        }
        double weight = 3.0 / 8.0 * h * ratio;

        data->lambda_tbl[i] = lambda;
        for (int k = 0; k < 3; ++k) {
            for (int j = 0; j < 3; ++j) {
                data->rgb_tbl[k][i] += data->xyz_to_rgb[k][j] * xyz[j] * I * weight;
            }
        }

        for (int k = 0; k < 3; ++k) {
            data->xyz_whitepoint[k] += xyz[k] * I * weight;
        }
    }
}

void eval_residual(const RGBtoSpectrumBuffer *data, const double *coeffs, const double *rgb,
                   double *residual) {
    double out[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < CIE_FINE_SAMPLES; ++i) {
        /* Scale lambda to 0..1 range */
        double lambda = (data->lambda_tbl[i] - CIE_LAMBDA_MIN) / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);

        /* Polynomial */
        double x = 0.0;
        for (int k = 0; k < 3; ++k) {
            x = x * lambda + coeffs[k];
        }

        /* Sigmoid */
        double s = sigmoid(x);

        /* Integrate against precomputed curves */
        for (int j = 0; j < 3; ++j) {
            out[j] += data->rgb_tbl[j][i] * s;
        }
    }

    cie_lab(out, data);
    memcpy(residual, rgb, sizeof(double) * 3);
    cie_lab(residual, data);

    for (int j = 0; j < 3; ++j) {
        residual[j] -= out[j];
    }
}

void eval_jacobian(const RGBtoSpectrumBuffer *data, const double *coeffs, const double *rgb,
                   double **jac) {
    double r0[3], r1[3], tmp[3];

    for (int i = 0; i < 3; ++i) {
        memcpy(tmp, coeffs, sizeof(double) * 3);
        tmp[i] -= RGB2SPEC_EPSILON;
        eval_residual(data, tmp, rgb, r0);

        memcpy(tmp, coeffs, sizeof(double) * 3);
        tmp[i] += RGB2SPEC_EPSILON;
        eval_residual(data, tmp, rgb, r1);

        for (int j = 0; j < 3; ++j) {
            jac[j][i] = (r1[j] - r0[j]) * 1.0 / (2 * RGB2SPEC_EPSILON);
        }
    }
}

void gauss_newton(const RGBtoSpectrumBuffer *data, const double rgb[3], double coeffs[3],
                  int it = 15) {
    for (int i = 0; i < it; ++i) {
        double J0[3], J1[3], J2[3], *J[3] = {J0, J1, J2};

        double residual[3];

        eval_residual(data, coeffs, rgb, residual);
        eval_jacobian(data, coeffs, rgb, J);

        int P[4];
        int rv = LUPDecompose(J, 3, 1e-15, P);
        if (rv != 1) {
            std::cout << "RGB " << rgb[0] << " " << rgb[1] << " " << rgb[2] << std::endl;
            std::cout << "-> " << coeffs[0] << " " << coeffs[1] << " " << coeffs[2] << std::endl;
            throw std::runtime_error("LU decomposition failed!");
        }

        double x[3];
        LUPSolve(J, P, residual, 3, x);

        double r = 0.0;
        for (int j = 0; j < 3; ++j) {
            coeffs[j] -= x[j];
            r += residual[j] * residual[j];
        }

        double max = std::max(std::max(coeffs[0], coeffs[1]), coeffs[2]);
        if (max > 200) {
            for (int j = 0; j < 3; ++j) {
                coeffs[j] *= 200 / max;
            }
        }

        if (r < 1e-6) {
            break;
        }
    }
}

void compute(double *out, int j, const RGBtoSpectrumBuffer &data, const double *scale, int l) {
    const double y = j / double(RES - 1);

    const double c0 = CIE_LAMBDA_MIN;
    const double c1 = 1.0 / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);
    const int start = RES / 5;

    for (int i = 0; i < RES; ++i) {
        const double x = i / double(RES - 1);
        double coeffs[3];
        double rgb[3];
        memset(coeffs, 0, sizeof(double) * 3);

        for (int k = start; k < RES; ++k) {
            double b = scale[k];

            rgb[l] = b;
            rgb[(l + 1) % 3] = x * b;
            rgb[(l + 2) % 3] = y * b;

            gauss_newton(&data, rgb, coeffs);

            double A = coeffs[0];
            double B = coeffs[1];
            double C = coeffs[2];

            int idx = ((l * RES + k) * RES + j) * RES + i;

            out[3 * idx + 0] = double(A * (sqr(c1)));
            out[3 * idx + 1] = double(B * c1 - 2 * A * c0 * (sqr(c1)));
            out[3 * idx + 2] = double(C - B * c0 * c1 + A * (sqr(c0 * c1)));
        }

        memset(coeffs, 0, sizeof(double) * 3);
        for (int k = start; k >= 0; --k) {
            double b = scale[k];

            rgb[l] = b;
            rgb[(l + 1) % 3] = x * b;
            rgb[(l + 2) % 3] = y * b;

            gauss_newton(&data, rgb, coeffs);

            double A = coeffs[0];
            double B = coeffs[1];
            double C = coeffs[2];

            int idx = ((l * RES + k) * RES + j) * RES + i;

            out[3 * idx + 0] = double(A * (sqr(c1)));
            out[3 * idx + 1] = double(B * c1 - 2 * A * c0 * (sqr(c1)));
            out[3 * idx + 2] = double(C - B * c0 * c1 + A * (sqr(c0 * c1)));
        }
    }
}

PBRT_CPU_GPU
RGBSigmoidPolynomial RGBtoSpectrumTable::operator()(const RGB &rgb) const {
    // Handle uniform _rgb_ values
    if (rgb[0] == rgb[1] && rgb[1] == rgb[2]) {
        return RGBSigmoidPolynomial(0, 0, (rgb[0] - 0.5f) / std::sqrt(rgb[0] * (1.0 - rgb[0])));
    }

    // Find maximum component and compute remapped component values
    int max_component =
        (rgb[0] > rgb[1]) ? ((rgb[0] > rgb[2]) ? 0 : 2) : ((rgb[1] > rgb[2]) ? 1 : 2);
    double z = rgb[max_component];
    double x = rgb[(max_component + 1) % 3] * (RES - 1) / z;
    double y = rgb[(max_component + 2) % 3] * (RES - 1) / z;

    // Compute integer indices and offsets for coefficient interpolation
    int xi = std::min((int)x, RES - 2);
    int yi = std::min((int)y, RES - 2);
    int zi = find_interval(RES, [&](int i) { return z_nodes[i] < z; });

    double dx = x - xi;
    double dy = y - yi;
    double dz = (z - z_nodes[zi]) / (z_nodes[zi + 1] - z_nodes[zi]);

    // Trilinearly interpolate sigmoid polynomial coefficients _c_
    double c[3];
    for (int i = 0; i < 3; ++i) {
        // Define _co_ lambda for looking up sigmoid polynomial coefficients
        auto co = [&](int _dx, int _dy, int _dz) {
            return coefficients[max_component][zi + _dz][yi + _dy][xi + _dx][i];
        };

        c[i] = lerp(
            dz, lerp(dy, lerp(dx, co(0, 0, 0), co(1, 0, 0)), lerp(dx, co(0, 1, 0), co(1, 1, 0))),
            lerp(dy, lerp(dx, co(0, 0, 1), co(1, 0, 1)), lerp(dx, co(0, 1, 1), co(1, 1, 1))));
    }

    return RGBSigmoidPolynomial(c[0], c[1], c[2]);
}

void RGBtoSpectrumTable::init(const std::string &str_gamut, ThreadPool &thread_pool) {
    if (str_gamut != "sRGB") {
        throw std::runtime_error("compute_spectrum_table_data: only sRGB is implemented");
    }

    Gamut gamut = Gamut::sRGB;

    RGBtoSpectrumBuffer rgb_to_spectrum_buffer;
    init_tables(&rgb_to_spectrum_buffer, gamut);

    for (int k = 0; k < RES; k++) {
        this->z_nodes[k] = smoothstep(smoothstep(double(k) / double(RES - 1)));
    }

    auto coefficients_ptr = (double *)this->coefficients;
    auto z_nodes_ptr = this->z_nodes;

    for (int l = 0; l < 3; ++l) {
        thread_pool.parallel_execute(
            0, RES, [coefficients_ptr, z_nodes_ptr, &rgb_to_spectrum_buffer, l](int j) {
                compute(coefficients_ptr, j, rgb_to_spectrum_buffer, z_nodes_ptr, l);
            });
    }
}
} // namespace RGBtoSpectrumData
