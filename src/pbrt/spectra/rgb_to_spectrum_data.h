#pragma once

#include <thread>
#include <mutex>
#include <stack>
#include <vector>

#include "pbrt/spectra/rgb.h"
#include "pbrt/spectra/rgb_sigmoid_polynomial.h"

namespace RGBtoSpectrumData {

static constexpr int CIE_SAMPLES = 95;

static constexpr int CIE_FINE_SAMPLES = (CIE_SAMPLES - 1) * 3 + 1;

static constexpr double RGB2SPEC_EPSILON = 1e-4;

static constexpr double CIE_LAMBDA_MIN = 360;
static constexpr double CIE_LAMBDA_MAX = 830;

static constexpr int RES = 64;

enum class Gamut {
    srgb,
    ProPhotoRGB,
    ACES2065_1,
    REC2020,
    ERGB,
    XYZ,
    DCI_P3,
    NO_GAMUT,
};

struct RGBtoSpectrumBuffer {
    double lambda_tbl[CIE_FINE_SAMPLES];
    double rgb_tbl[3][CIE_FINE_SAMPLES];
    double rgb_to_xyz[3][3];
    double xyz_to_rgb[3][3];
    double xyz_whitepoint[3];
};

struct RGBtoSpectrumTableCPU {
    double *z_nodes;
    double *coefficients;

    RGBtoSpectrumTableCPU()
        : z_nodes(new double[RES]), coefficients(new double[3 * RES * RES * RES * 3]) {}

    ~RGBtoSpectrumTableCPU() {
        delete[] z_nodes;
        delete[] coefficients;
    }
};

struct RGBtoSpectrumTableGPU {
    double z_nodes[RES];
    double coefficients[3][RES][RES][RES][3];

    PBRT_GPU
    RGBSigmoidPolynomial operator()(const RGB &rgb) const {
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
                dz,
                lerp(dy, lerp(dx, co(0, 0, 0), co(1, 0, 0)), lerp(dx, co(0, 1, 0), co(1, 1, 0))),
                lerp(dy, lerp(dx, co(0, 0, 1), co(1, 0, 1)), lerp(dx, co(0, 1, 1), co(1, 1, 1))));
        }

        return RGBSigmoidPolynomial(c[0], c[1], c[2]);
    }
};

// clang-format off
const double cie_x[CIE_SAMPLES] = {
    0.000129900000, 0.000232100000, 0.000414900000, 0.000741600000, 0.001368000000,
    0.002236000000, 0.004243000000, 0.007650000000, 0.014310000000, 0.023190000000,
    0.043510000000, 0.077630000000, 0.134380000000, 0.214770000000, 0.283900000000,
    0.328500000000, 0.348280000000, 0.348060000000, 0.336200000000, 0.318700000000,
    0.290800000000, 0.251100000000, 0.195360000000, 0.142100000000, 0.095640000000,
    0.057950010000, 0.032010000000, 0.014700000000, 0.004900000000, 0.002400000000,
    0.009300000000, 0.029100000000, 0.063270000000, 0.109600000000, 0.165500000000,
    0.225749900000, 0.290400000000, 0.359700000000, 0.433449900000, 0.512050100000,
    0.594500000000, 0.678400000000, 0.762100000000, 0.842500000000, 0.916300000000,
    0.978600000000, 1.026300000000, 1.056700000000, 1.062200000000, 1.045600000000,
    1.002600000000, 0.938400000000, 0.854449900000, 0.751400000000, 0.642400000000,
    0.541900000000, 0.447900000000, 0.360800000000, 0.283500000000, 0.218700000000,
    0.164900000000, 0.121200000000, 0.087400000000, 0.063600000000, 0.046770000000,
    0.032900000000, 0.022700000000, 0.015840000000, 0.011359160000, 0.008110916000,
    0.005790346000, 0.004109457000, 0.002899327000, 0.002049190000, 0.001439971000,
    0.000999949300, 0.000690078600, 0.000476021300, 0.000332301100, 0.000234826100,
    0.000166150500, 0.000117413000, 0.000083075270, 0.000058706520, 0.000041509940,
    0.000029353260, 0.000020673830, 0.000014559770, 0.000010253980, 0.000007221456,
    0.000005085868, 0.000003581652, 0.000002522525, 0.000001776509, 0.000001251141,
};

const double cie_y[CIE_SAMPLES] = {
    0.000003917000, 0.000006965000, 0.000012390000, 0.000022020000, 0.000039000000,
    0.000064000000, 0.000120000000, 0.000217000000, 0.000396000000, 0.000640000000,
    0.001210000000, 0.002180000000, 0.004000000000, 0.007300000000, 0.011600000000,
    0.016840000000, 0.023000000000, 0.029800000000, 0.038000000000, 0.048000000000,
    0.060000000000, 0.073900000000, 0.090980000000, 0.112600000000, 0.139020000000,
    0.169300000000, 0.208020000000, 0.258600000000, 0.323000000000, 0.407300000000,
    0.503000000000, 0.608200000000, 0.710000000000, 0.793200000000, 0.862000000000,
    0.914850100000, 0.954000000000, 0.980300000000, 0.994950100000, 1.000000000000,
    0.995000000000, 0.978600000000, 0.952000000000, 0.915400000000, 0.870000000000,
    0.816300000000, 0.757000000000, 0.694900000000, 0.631000000000, 0.566800000000,
    0.503000000000, 0.441200000000, 0.381000000000, 0.321000000000, 0.265000000000,
    0.217000000000, 0.175000000000, 0.138200000000, 0.107000000000, 0.081600000000,
    0.061000000000, 0.044580000000, 0.032000000000, 0.023200000000, 0.017000000000,
    0.011920000000, 0.008210000000, 0.005723000000, 0.004102000000, 0.002929000000,
    0.002091000000, 0.001484000000, 0.001047000000, 0.000740000000, 0.000520000000,
    0.000361100000, 0.000249200000, 0.000171900000, 0.000120000000, 0.000084800000,
    0.000060000000, 0.000042400000, 0.000030000000, 0.000021200000, 0.000014990000,
    0.000010600000, 0.000007465700, 0.000005257800, 0.000003702900, 0.000002607800,
    0.000001836600, 0.000001293400, 0.000000910930, 0.000000641530, 0.000000451810,
};

const double cie_z[CIE_SAMPLES] = {
    0.000606100000, 0.001086000000, 0.001946000000, 0.003486000000, 0.006450001000,
    0.010549990000, 0.020050010000, 0.036210000000, 0.067850010000, 0.110200000000,
    0.207400000000, 0.371300000000, 0.645600000000, 1.039050100000, 1.385600000000,
    1.622960000000, 1.747060000000, 1.782600000000, 1.772110000000, 1.744100000000,
    1.669200000000, 1.528100000000, 1.287640000000, 1.041900000000, 0.812950100000,
    0.616200000000, 0.465180000000, 0.353300000000, 0.272000000000, 0.212300000000,
    0.158200000000, 0.111700000000, 0.078249990000, 0.057250010000, 0.042160000000,
    0.029840000000, 0.020300000000, 0.013400000000, 0.008749999000, 0.005749999000,
    0.003900000000, 0.002749999000, 0.002100000000, 0.001800000000, 0.001650001000,
    0.001400000000, 0.001100000000, 0.001000000000, 0.000800000000, 0.000600000000,
    0.000340000000, 0.000240000000, 0.000190000000, 0.000100000000, 0.000049999990,
    0.000030000000, 0.000020000000, 0.000010000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
};

#define N(x) (x / 10566.864005283874576)
const double cie_d65[CIE_SAMPLES] = {
    N(46.6383), N(49.3637), N(52.0891), N(51.0323), N(49.9755), N(52.3118), N(54.6482),
    N(68.7015), N(82.7549), N(87.1204), N(91.486),  N(92.4589), N(93.4318), N(90.057),
    N(86.6823), N(95.7736), N(104.865), N(110.936), N(117.008), N(117.41),  N(117.812),
    N(116.336), N(114.861), N(115.392), N(115.923), N(112.367), N(108.811), N(109.082),
    N(109.354), N(108.578), N(107.802), N(106.296), N(104.79),  N(106.239), N(107.689),
    N(106.047), N(104.405), N(104.225), N(104.046), N(102.023), N(100.0),   N(98.1671),
    N(96.3342), N(96.0611), N(95.788),  N(92.2368), N(88.6856), N(89.3459), N(90.0062),
    N(89.8026), N(89.5991), N(88.6489), N(87.6987), N(85.4936), N(83.2886), N(83.4939),
    N(83.6992), N(81.863),  N(80.0268), N(80.1207), N(80.2146), N(81.2462), N(82.2778),
    N(80.281),  N(78.2842), N(74.0027), N(69.7213), N(70.6652), N(71.6091), N(72.979),
    N(74.349),  N(67.9765), N(61.604),  N(65.7448), N(69.8856), N(72.4863), N(75.087),
    N(69.3398), N(63.5927), N(55.0054), N(46.4182), N(56.6118), N(66.8054), N(65.0941),
    N(63.3828), N(63.8434), N(64.304),  N(61.8779), N(59.4519), N(55.7054), N(51.959),
    N(54.6998), N(57.4406), N(58.8765), N(60.3125),
};
#undef N

#define N(x) (x / 106.8)
const double cie_e[CIE_SAMPLES] = {
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
    N(1.0), N(1.0), N(1.0), N(1.0), N(1.0),
};
#undef N

#define N(x) (x / 10503.2)
const double cie_d50[CIE_SAMPLES] = {
    N(23.942000),  N(25.451000),  N(26.961000),  N(25.724000),  N(24.488000),
    N(27.179000),  N(29.871000),  N(39.589000),  N(49.308000),  N(52.910000),
    N(56.513000),  N(58.273000),  N(60.034000),  N(58.926000),  N(57.818000),
    N(66.321000),  N(74.825000),  N(81.036000),  N(87.247000),  N(88.930000),
    N(90.612000),  N(90.990000),  N(91.368000),  N(93.238000),  N(95.109000),
    N(93.536000),  N(91.963000),  N(93.843000),  N(95.724000),  N(96.169000),
    N(96.613000),  N(96.871000),  N(97.129000),  N(99.614000),  N(102.099000),
    N(101.427000), N(100.755000), N(101.536000), N(102.317000), N(101.159000),
    N(100.000000), N(98.868000),  N(97.735000),  N(98.327000),  N(98.918000),
    N(96.208000),  N(93.499000),  N(95.593000),  N(97.688000),  N(98.478000),
    N(99.269000),  N(99.155000),  N(99.042000),  N(97.382000),  N(95.722000),
    N(97.290000),  N(98.857000),  N(97.262000),  N(95.667000),  N(96.929000),
    N(98.190000),  N(100.597000), N(103.003000), N(101.068000), N(99.133000),
    N(93.257000),  N(87.381000),  N(89.492000),  N(91.604000),  N(92.246000),
    N(92.889000),  N(84.872000),  N(76.854000),  N(81.683000),  N(86.511000),
    N(89.546000),  N(92.580000),  N(85.405000),  N(78.230000),  N(67.961000),
    N(57.692000),  N(70.307000),  N(82.923000),  N(80.599000),  N(78.274000),
    N(0),          N(0),          N(0),          N(0),          N(0),
    N(0),          N(0),          N(0),          N(0),
};
#undef N
// clang-format on

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

const double xyz_to_xyz[3][3] = {
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0},
};

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

double sigmoid(double x) {
    return 0.5 * x / std::sqrt(1.0 + x * x) + 0.5;
}

double smoothstep(double x) {
    return x * x * (3.0 - 2.0 * x);
}

double sqr(double x) {
    return x * x;
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
    case Gamut::srgb: {
        illuminant = cie_d65;
        memcpy(data->xyz_to_rgb, xyz_to_srgb, sizeof(double) * 9);
        memcpy(data->rgb_to_xyz, srgb_to_xyz, sizeof(double) * 9);
        break;
    }

    default: {
        throw std::runtime_error("init_tables(): invalid/unsupported gamut.");
    }
    }

    for (int i = 0; i < CIE_FINE_SAMPLES; ++i) {
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
        for (int i = 0; i < 3; ++i) {
            data->xyz_whitepoint[i] += xyz[i] * I * weight;
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

RGBtoSpectrumTableCPU compute_spectrum_table_data(const std::string &str_gamut) {
    if (str_gamut != "sRGB") {
        throw std::runtime_error("compute_spectrum_table_data: only sRGB is implemented");
    }

    Gamut gamut = Gamut::srgb;

    RGBtoSpectrumBuffer data;
    init_tables(&data, gamut);

    RGBtoSpectrumTableCPU result;

    for (int k = 0; k < RES; k++) {
        result.z_nodes[k] = smoothstep(smoothstep(k / double(RES - 1)));
    }

    for (int l = 0; l < 3; ++l) {
        thread_pool->parallel_execute(0, RES, [&result, &data, l](int j) {
            compute(result.coefficients, j, data, result.z_nodes, l);
        });
    }

    return result;
}
}; // namespace RGBtoSpectrumData
