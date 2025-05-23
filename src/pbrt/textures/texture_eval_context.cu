#include <pbrt/base/interaction.h>
#include <pbrt/textures/texture_eval_context.h>

PBRT_CPU_GPU
TextureEvalContext::TextureEvalContext(const Interaction &intr) : p(intr.p()), uv(intr.uv) {}

PBRT_CPU_GPU
TextureEvalContext::TextureEvalContext(const SurfaceInteraction &si)
    : p(si.p()), dpdx(si.dpdx), dpdy(si.dpdy), n(si.n), uv(si.uv), dudx(si.dudx), dudy(si.dudy),
      dvdx(si.dvdx), dvdy(si.dvdy), faceIndex(si.faceIndex) {}

PBRT_CPU_GPU
MaterialEvalContext::MaterialEvalContext(const SurfaceInteraction &si)
    : TextureEvalContext(si), wo(si.wo), ns(si.shading.n), dpdus(si.shading.dpdu) {}
