#include <pbrt/base/interaction.h>
#include <pbrt/base/shape.h>
#include <pbrt/shapes/triangle.h>

PBRT_CPU_GPU
pbrt::optional<ShapeIntersection> Triangle::intersect(const Ray &ray, FloatType t_max) const {
    Point3f points[3];
    get_points(points);

    auto tri_intersection = intersect_triangle(ray, t_max, points[0], points[1], points[2]);
    if (!tri_intersection) {
        return {};
    }

    SurfaceInteraction si = interaction_from_intersection(tri_intersection.value(), -ray.d);

    return ShapeIntersection(si, tri_intersection->t);
}

PBRT_CPU_GPU
FloatType Triangle::pdf(const Interaction &in) const {
    return 1.0 / this->area();
}

PBRT_CPU_GPU
FloatType Triangle::pdf(const ShapeSampleContext &ctx, const Vector3f &wi) const {
    FloatType solidAngle = solid_angle(ctx.p());
    // Return PDF based on uniform area sampling for challenging triangles
    if (solidAngle < MinSphericalSampleArea || solidAngle > MaxSphericalSampleArea) {
        // Intersect sample ray with shape geometry
        Ray ray = ctx.spawn_ray(wi);

        auto isect = intersect(ray, Infinity);
        if (!isect) {
            return 0;
        }

        // Compute PDF in solid angle measure from shape intersection point
        FloatType pdf = (1 / area()) / (isect->interaction.n.abs_dot(-wi) /
                                        ctx.p().squared_distance(isect->interaction.p()));

        if (isinf(pdf)) {
            pdf = 0;
        }

        return pdf;
    }

    FloatType pdf = 1 / solidAngle;
    // Adjust PDF for warp product sampling of triangle $\cos\theta$ factor
    if (ctx.ns != Normal3f(0, 0, 0)) {
        // Get triangle vertices in _p0_, _p1_, and _p2_
        /*
        const TriangleMesh *mesh = GetMesh();
        const int *v = &mesh->vertexIndices[3 * triIndex];


        Point3f p0 = mesh->p[v[0]], p1 = mesh->p[v[1]], p2 = mesh->p[v[2]];
        */
        Point3f points[3];
        get_points(points);
        const auto p0 = points[0];
        const auto p1 = points[1];
        const auto p2 = points[2];

        Point2f u = InvertSphericalTriangleSample(points, ctx.p(), wi);

        // Compute $\cos\theta$-based weights _w_ at sample domain corners
        Point3f rp = ctx.p();
        Vector3f wi[3] = {(p0 - rp).normalize(), (p1 - rp).normalize(), (p2 - rp).normalize()};

        FloatType w[4] = {std::max<FloatType>(0.01, ctx.ns.abs_dot(wi[1])),
                          std::max<FloatType>(0.01, ctx.ns.abs_dot(wi[1])),
                          std::max<FloatType>(0.01, ctx.ns.abs_dot(wi[0])),
                          std::max<FloatType>(0.01, ctx.ns.abs_dot(wi[2]))};

        pdf *= bilinear_pdf(u, w);
    }

    return pdf;
}

PBRT_CPU_GPU
pbrt::optional<ShapeSample> Triangle::sample(Point2f u) const {
    const int *v = &(mesh->vertex_indices[3 * triangle_idx]);
    const Point3f p0 = mesh->p[v[0]];
    const Point3f p1 = mesh->p[v[1]];
    const Point3f p2 = mesh->p[v[2]];

    FloatType b[3];
    sample_uniform_triangle(b, u);
    const Point3f p = b[0] * p0 + b[1] * p1 + b[2] * p2;

    // Compute surface normal for sampled point on triangle
    // Normal3f n = Normalize(Normal3f(Cross(p1 - p0, p2 - p0)));
    Normal3f n = Normal3f((p1 - p0).cross(p2 - p0).normalize());

    if (mesh->n) {
        Normal3f ns =
            (b[0] * mesh->n[v[0]] + b[1] * mesh->n[v[1]] + (1 - b[0] - b[1]) * mesh->n[v[2]]);
        n = n.face_forward(ns);
    } else if ((mesh->reverse_orientation ^ mesh->transformSwapsHandedness)) {
        // this part not implemented
        n *= -1;
    }

    Point2f uv[3];
    if (mesh->uv) {
        for (uint idx = 0; idx < 3; ++idx) {
            uv[idx] = mesh->uv[v[idx]];
        }
    } else {
        uv[0] = Point2f(0, 0);
        uv[1] = Point2f(1, 0);
        uv[2] = Point2f(1, 1);
    }

    Point2f uvSample = b[0] * uv[0] + b[1] * uv[1] + b[2] * uv[2];

    // Compute error bounds _pError_ for sampled point on triangle
    Point3f pAbsSum = (b[0] * p0).abs() + (b[1] * p1).abs() + ((1 - b[0] - b[1]) * p2).abs();

    Vector3f pError = (gamma(6) * pAbsSum).to_vector3();

    return ShapeSample{
        .interaction = Interaction(Point3fi(p, pError), n, uvSample),
        .pdf = FloatType(1.0) / area(),
    };
}

PBRT_CPU_GPU
pbrt::optional<ShapeSample> Triangle::sample(const ShapeSampleContext &ctx, Point2f u) const {
    Point3f points[3];
    get_points(points);
    const auto p0 = points[0];
    const auto p1 = points[1];
    const auto p2 = points[2];
    const int *v = &(mesh->vertex_indices[3 * triangle_idx]);

    // Use uniform area sampling for numerically unstable cases
    FloatType solid_angle = this->solid_angle(ctx.p());

    if (solid_angle < MinSphericalSampleArea || solid_angle > MaxSphericalSampleArea) {
        auto ss = sample(u);
        Vector3f wi = ss->interaction.p() - ctx.p();
        if (wi.squared_length() == 0) {
            return {};
        }
        wi = wi.normalize();

        // Convert area sampling PDF in _ss_ to solid angle measure
        ss->pdf /=
            ss->interaction.n.abs_dot(-wi) / (ctx.p() - ss->interaction.p()).squared_length();

        if (is_inf(ss->pdf)) {
            return {};
        }

        return ss;
    }

    // Sample spherical triangle from reference point
    // Apply warp product sampling for cosine factor at reference point
    FloatType pdf = 1.0;
    if (ctx.ns != Normal3f(0, 0, 0)) {
        // Compute $\cos\theta$-based weights _w_ at sample domain corners
        Point3f rp = ctx.p();
        Vector3f wi[3] = {(p0 - rp).normalize(), (p1 - rp).normalize(), (p2 - rp).normalize()};

        FloatType w[4] = {std::max<FloatType>(0.01, ctx.ns.abs_dot(wi[1])),
                          std::max<FloatType>(0.01, ctx.ns.abs_dot(wi[1])),
                          std::max<FloatType>(0.01, ctx.ns.abs_dot(wi[0])),
                          std::max<FloatType>(0.01, ctx.ns.abs_dot(wi[2]))};
        u = sample_bilinear(u, w);
        pdf = bilinear_pdf(u, w);
    }

    FloatType triPDF;
    FloatType b[3];
    sample_spherical_triangle(b, &triPDF, points, ctx.p(), u);
    if (triPDF == 0.0) {
        return {};
    }
    pdf *= triPDF;

    // Compute error bounds _pError_ for sampled point on triangle
    Point3f pAbsSum = (b[0] * p0).abs() + (b[1] * p1).abs() + ((1 - b[0] - b[1]) * p2).abs();
    Vector3f pError = (gamma(6) * pAbsSum).to_vector3();

    // Return _ShapeSample_ for solid angle sampled point on triangle
    Point3f p = b[0] * p0 + b[1] * p1 + b[2] * p2;
    // Compute surface normal for sampled point on triangle
    Normal3f n = Normal3f((p1 - p0).cross(p2 - p0).normalize());

    if (mesh->n) {
        Normal3f ns(b[0] * mesh->n[v[0]] + b[1] * mesh->n[v[1]] +
                    (1 - b[0] - b[1]) * mesh->n[v[2]]);
        n = n.face_forward(ns);
    } else if (mesh->reverse_orientation ^ mesh->transformSwapsHandedness) {
        n *= -1;
    }

    Point2f uv[3];
    if (mesh->uv) {
        uv[0] = mesh->uv[v[0]];
        uv[1] = mesh->uv[v[1]];
        uv[2] = mesh->uv[v[2]];
    } else {
        uv[0] = Point2f(0, 0);
        uv[1] = Point2f(1, 0);
        uv[2] = Point2f(1, 1);
    }

    Point2f uvSample = b[0] * uv[0] + b[1] * uv[1] + b[2] * uv[2];

    return ShapeSample{
        .interaction = Interaction(Point3fi(p, pError), n, uvSample),
        .pdf = pdf,
    };
}

PBRT_CPU_GPU
pbrt::optional<Triangle::TriangleIntersection>
Triangle::intersect_triangle(const Ray &ray, FloatType t_max, const Point3f &p0, const Point3f &p1,
                             const Point3f &p2) const {
    // Return no intersection if triangle is degenerate
    if ((p2 - p0).cross(p1 - p0).squared_length() == 0.0) {
        return {};
    }

    // Transform triangle vertices to ray coordinate space
    // Translate vertices based on ray origin
    Point3f p0t = p0 - ray.o.to_vector3();
    Point3f p1t = p1 - ray.o.to_vector3();
    Point3f p2t = p2 - ray.o.to_vector3();

    // Permute components of triangle vertices and ray direction
    uint8_t kz = ray.d.abs().max_component_index();
    uint8_t kx = (kz + 1) % 3;
    uint8_t ky = (kz + 2) % 3;

    uint8_t permuted_idx[3] = {kx, ky, kz};
    Vector3f d = ray.d.permute(permuted_idx);

    p0t = p0t.permute(permuted_idx);
    p1t = p1t.permute(permuted_idx);
    p2t = p2t.permute(permuted_idx);

    // Apply shear transformation to translated vertex positions
    FloatType Sx = -d.x / d.z;
    FloatType Sy = -d.y / d.z;
    FloatType Sz = 1 / d.z;
    p0t.x += Sx * p0t.z;
    p0t.y += Sy * p0t.z;
    p1t.x += Sx * p1t.z;
    p1t.y += Sy * p1t.z;
    p2t.x += Sx * p2t.z;
    p2t.y += Sy * p2t.z;

    // Compute edge function coefficients _e0_, _e1_, and _e2_
    FloatType e0 = difference_of_products(p1t.x, p2t.y, p1t.y, p2t.x);
    FloatType e1 = difference_of_products(p2t.x, p0t.y, p2t.y, p0t.x);
    FloatType e2 = difference_of_products(p0t.x, p1t.y, p0t.y, p1t.x);

    // Fall back to double-precision test at triangle edges
    if (sizeof(FloatType) == sizeof(float) && (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f)) {
        double p2txp1ty = (double)p2t.x * (double)p1t.y;
        double p2typ1tx = (double)p2t.y * (double)p1t.x;
        e0 = (FloatType)(p2typ1tx - p2txp1ty);
        double p0txp2ty = (double)p0t.x * (double)p2t.y;
        double p0typ2tx = (double)p0t.y * (double)p2t.x;
        e1 = (FloatType)(p0typ2tx - p0txp2ty);
        double p1txp0ty = (double)p1t.x * (double)p0t.y;
        double p1typ0tx = (double)p1t.y * (double)p0t.x;
        e2 = (FloatType)(p1typ0tx - p1txp0ty);
    }

    // Perform triangle edge and determinant tests
    if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0)) {
        return {};
    }

    FloatType det = e0 + e1 + e2;
    if (det == 0) {
        return {};
    }

    // Compute scaled hit distance to triangle and test against ray $t$ range
    p0t.z *= Sz;
    p1t.z *= Sz;
    p2t.z *= Sz;
    FloatType tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
    if (det < 0 && (tScaled >= 0 || tScaled < t_max * det)) {
        return {};
    }

    if (det > 0 && (tScaled <= 0 || tScaled > t_max * det)) {
        return {};
    }

    // Compute barycentric coordinates and $t$ value for triangle intersection
    FloatType invDet = 1 / det;
    FloatType b0 = e0 * invDet, b1 = e1 * invDet, b2 = e2 * invDet;
    FloatType t = tScaled * invDet;

    // Ensure that computed triangle $t$ is conservatively greater than zero
    // Compute $\delta_z$ term for triangle $t$ error bounds
    // FloatType maxZt = MaxComponentValue(Abs(Vector3f(p0t.z, p1t.z, p2t.z)));

    FloatType maxZt = Vector3f(p0t.z, p1t.z, p2t.z).abs().max_component_value();
    FloatType deltaZ = gamma(3) * maxZt;

    // Compute $\delta_x$ and $\delta_y$ terms for triangle $t$ error bounds
    FloatType maxXt = Vector3f(p0t.x, p1t.x, p2t.x).abs().max_component_value();
    FloatType maxYt = Vector3f(p0t.y, p1t.y, p2t.y).abs().max_component_value();

    FloatType deltaX = gamma(5) * (maxXt + maxZt);
    FloatType deltaY = gamma(5) * (maxYt + maxZt);

    // Compute $\delta_e$ term for triangle $t$ error bounds
    FloatType deltaE = 2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

    // Compute $\delta_t$ term for triangle $t$ error bounds and check _t_
    FloatType maxE = Vector3f(e0, e1, e2).abs().max_component_value();
    FloatType deltaT =
        3 * (gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) * std::abs(invDet);
    if (t <= deltaT) {
        return {};
    }

    // Return _TriangleIntersection_ for intersection
    return TriangleIntersection(b0, b1, b2, t);
}

PBRT_CPU_GPU
SurfaceInteraction Triangle::interaction_from_intersection(const TriangleIntersection &ti,
                                                           const Vector3f &wo) const {
    const int *v = &(mesh->vertex_indices[3 * triangle_idx]);
    const Point3f p0 = mesh->p[v[0]];
    const Point3f p1 = mesh->p[v[1]];
    const Point3f p2 = mesh->p[v[2]];

    // Compute triangle partial derivatives
    // Compute deltas and matrix determinant for triangle partial derivatives
    // Get triangle texture coordinates in _uv_ array

    Point2f uv[3];
    if (mesh->uv) {
        for (uint idx = 0; idx < 3; ++idx) {
            uv[idx] = mesh->uv[v[idx]];
        }
    } else {
        uv[0] = Point2f(0, 0);
        uv[1] = Point2f(1, 0);
        uv[2] = Point2f(1, 1);
    }

    Vector2f duv02 = uv[0] - uv[2], duv12 = uv[1] - uv[2];
    Vector3f dp02 = p0 - p2;
    Vector3f dp12 = p1 - p2;
    FloatType determinant = difference_of_products(duv02[0], duv12[1], duv02[1], duv12[0]);

    Vector3f dpdu, dpdv;
    bool degenerateUV = std::abs(determinant) < 1e-9f;
    if (!degenerateUV) {
        // Compute triangle $\dpdu$ and $\dpdv$ via matrix inversion
        FloatType invdet = 1 / determinant;
        dpdu = difference_of_products(duv12[1], dp02, duv02[1], dp12) * invdet;
        dpdv = difference_of_products(duv02[0], dp12, duv12[0], dp02) * invdet;
    }
    // Handle degenerate triangle $(u,v)$ parameterization or partial derivatives
    if (degenerateUV || dpdu.cross(dpdv).squared_length() == 0) {
        Vector3f ng = (p2 - p0).cross(p1 - p0);
        if (ng.squared_length() == 0) {
            ng = (p2 - p0).cross(p1 - p0);
        }
        ng.normalize().coordinate_system(&dpdu, &dpdv);
    }

    // Interpolate $(u,v)$ parametric coordinates and hit point
    Point3f pHit = ti.b0 * p0 + ti.b1 * p1 + ti.b2 * p2;
    Point2f uvHit = ti.b0 * uv[0] + ti.b1 * uv[1] + ti.b2 * uv[2];

    bool flipNormal = mesh->reverse_orientation ^ mesh->transformSwapsHandedness;
    // Compute error bounds _pError_ for triangle intersection
    Point3f pAbsSum = (ti.b0 * p0).abs() + (ti.b1 * p1).abs() + (ti.b2 * p2).abs();
    Vector3f pError = gamma(7) * pAbsSum.to_vector3();

    SurfaceInteraction isect(Point3fi(pHit, pError), uvHit, wo, dpdu, dpdv, Normal3f(), Normal3f(),
                             flipNormal);

    isect.faceIndex = mesh->faceIndices ? mesh->faceIndices[triangle_idx] : 0;

    // Set final surface normal and shading geometry for triangle
    // Override surface normal in _isect_ for triangle
    isect.n = Normal3f(dp02.cross(dp12).normalize());
    isect.shading.n = isect.n;

    if (mesh->reverse_orientation ^ mesh->transformSwapsHandedness) {
        isect.n = isect.shading.n = -isect.n;
    }

    if (mesh->n || mesh->s) {
        // Initialize _Triangle_ shading geometry
        // Compute shading normal _ns_ for triangle
        Normal3f ns;
        if (mesh->n) {
            ns = ti.b0 * mesh->n[v[0]] + ti.b1 * mesh->n[v[1]] + ti.b2 * mesh->n[v[2]];
            ns = ns.squared_length() > 0 ? ns.normalize() : isect.n;
        } else {
            ns = isect.n;
        }

        // Compute shading tangent _ss_ for triangle
        Vector3f ss;
        if (mesh->s) {
            ss = ti.b0 * mesh->s[v[0]] + ti.b1 * mesh->s[v[1]] + ti.b2 * mesh->s[v[2]];
            if (ss.squared_length() == 0) {
                ss = isect.dpdu;
            }
        } else {
            ss = isect.dpdu;
        }

        // Compute shading bitangent _ts_ for triangle and adjust _ss_
        auto ts = ns.to_vector3().cross(ss);
        if (ts.squared_length() > 0) {
            ss = ts.cross(ns.to_vector3());
        } else {
            ns.to_vector3().coordinate_system(&ss, &ts);
        }

        // Compute $\dndu$ and $\dndv$ for triangle shading geometry
        Normal3f dndu, dndv;
        if (mesh->n) {
            // Compute deltas for triangle partial derivatives of normal
            Vector2f duv02 = uv[0] - uv[2];
            Vector2f duv12 = uv[1] - uv[2];
            Normal3f dn1 = mesh->n[v[0]] - mesh->n[v[2]];
            Normal3f dn2 = mesh->n[v[1]] - mesh->n[v[2]];

            auto determinant = difference_of_products(duv02[0], duv12[1], duv02[1], duv12[0]);
            bool degenerateUV = std::abs(determinant) < 1e-9;
            if (degenerateUV) {
                // We can still compute dndu and dndv, with respect to the
                // same arbitrary coordinate system we use to compute dpdu
                // and dpdv when this happens. It's important to do this
                // (rather than giving up) so that ray differentials for
                // rays reflected from triangles with degenerate
                // parameterizations are still reasonable.

                auto dn = (mesh->n[v[2]] - mesh->n[v[0]]).cross(mesh->n[v[1]] - mesh->n[v[0]]);

                if (dn.squared_length() == 0) {
                    dndu = dndv = Normal3f(0, 0, 0);
                } else {
                    Vector3f dnu, dnv;
                    dn.coordinate_system(&dnu, &dnv);
                    dndu = Normal3f(dnu);
                    dndv = Normal3f(dnv);
                }
            } else {
                auto invDet = 1 / determinant;
                dndu = difference_of_products(duv12[1], dn1, duv02[1], dn2) * invDet;
                dndv = difference_of_products(duv02[0], dn2, duv12[0], dn1) * invDet;
            }
        } else {
            dndu = dndv = Normal3f(0, 0, 0);
        }

        isect.set_shading_geometry(ns, ss, ts, dndu, dndv, true);
    }

    return isect;
}
