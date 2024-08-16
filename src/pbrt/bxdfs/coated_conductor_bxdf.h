#pragma once

#include "pbrt/bxdfs/conductor_bxdf.h"
#include "pbrt/bxdfs/dielectric_bxdf.h"
#include "pbrt/bxdfs/layered_bxdf.h"

class CoatedConductorBxDF : public LayeredBxDF<DielectricBxDF, ConductorBxDF, true> {
  public:
    using LayeredBxDF::LayeredBxDF;
};
