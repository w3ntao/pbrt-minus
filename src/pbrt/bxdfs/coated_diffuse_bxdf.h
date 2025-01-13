#pragma once

#include <pbrt/bxdfs/dielectric_bxdf.h>
#include <pbrt/bxdfs/diffuse_bxdf.h>
#include <pbrt/bxdfs/layered_bxdf.h>

class CoatedDiffuseBxDF : public LayeredBxDF<DielectricBxDF, DiffuseBxDF, true> {
  public:
    using LayeredBxDF::LayeredBxDF;
};
