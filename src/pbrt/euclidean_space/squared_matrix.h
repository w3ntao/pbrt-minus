#pragma once

template <int N>
class SquareMatrix {
  private:
    double m[N][N];

  public:
    PBRT_CPU_GPU SquareMatrix(double data[N][N]) {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                m[i][k] = data[i][k];
            }
        }
    }

    PBRT_CPU_GPU const double *operator[](int i) const {
        return m[i];
    }
};
