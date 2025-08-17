#pragma once

#include <pbrt/distribution/piecewise_constant_1d.h>
#include <pbrt/euclidean_space/bounds2.h>
#include <pbrt/util/array_2d.h>

class PiecewiseConstant2D {
  public:
    PiecewiseConstant2D() = default;

    PiecewiseConstant2D(const Array2D<Real> *func, const Bounds2f &_domain,
                        GPUMemoryAllocator &allocator)
        : domain(_domain) {
        const int nu = func->x_size();
        const int nv = func->y_size();

        pConditionalV = allocator.allocate<PiecewiseConstant1D>(nv);
        for (int v = 0; v < nv; ++v) {
            // Compute conditional sampling distribution for $\tilde{v}$
            pConditionalV[v] = PiecewiseConstant1D(func->get_values_ptr() + v * nu, nu,
                                                   domain.p_min[0], domain.p_max[0], allocator);
        }

        std::vector<Real> marginalFunc;
        for (int v = 0; v < nv; ++v) {
            marginalFunc.push_back(pConditionalV[v].integral());
        }

        pMarginal = PiecewiseConstant1D(marginalFunc.data(), marginalFunc.size(), domain.p_min[1],
                                        domain.p_max[1], allocator);
    }

    PBRT_CPU_GPU
    Point2f sample(Point2f u, Real *pdf = nullptr, Point2i *offset = nullptr) const {
        Real pdfs[2];
        Point2i uv;
        Real d1 = pMarginal.sample(u[1], &pdfs[1], &uv[1]);
        Real d0 = pConditionalV[uv[1]].sample(u[0], &pdfs[0], &uv[0]);

        if (pdf) {
            *pdf = pdfs[0] * pdfs[1];
        }

        if (offset) {
            *offset = uv;
        }

        return Point2f(d0, d1);
    }

  private:
    Bounds2f domain;
    PiecewiseConstant1D *pConditionalV = nullptr;
    PiecewiseConstant1D pMarginal;
};
