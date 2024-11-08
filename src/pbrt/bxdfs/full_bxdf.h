#pragma once

#include "pbrt/bxdfs/coated_conductor_bxdf.h"
#include "pbrt/bxdfs/coated_diffuse_bxdf.h"
#include "pbrt/bxdfs/conductor_bxdf.h"
#include "pbrt/bxdfs/dielectric_bxdf.h"
#include "pbrt/bxdfs/diffuse_bxdf.h"

struct FullBxDF {
    CoatedConductorBxDF coated_conductor_bxdf;
    CoatedDiffuseBxDF coated_diffuse_bxdf;
    ConductorBxDF conductor_bxdf;
    DielectricBxDF dielectric_bxdf;
    DiffuseBxDF diffuse_bxdf;
};
