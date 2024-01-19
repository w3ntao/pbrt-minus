#pragma once

struct Intersection;

#include "base/ray.h"
#include "base/shape.h"

PBRT_GPU double schlick(double cosine, double ref_idx) {
    double r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

PBRT_GPU bool refract(const Vector3f &v, const Vector3f &n, double ni_over_nt,
                      Vector3f &refracted) {
    Vector3f uv = v.normalize();
    double dt = uv.dot(n);
    double discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);

    if (discriminant <= 0) {
        return false;
    }

    refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
    return true;
}

PBRT_GPU Vector3f random_vector(curandState *local_rand_state) {
    return Vector3f(curand_uniform(local_rand_state), curand_uniform(local_rand_state),
                    curand_uniform(local_rand_state));
}

PBRT_GPU Vector3f random_in_unit_sphere(curandState *local_rand_state) {
    Vector3f p;
    do {
        p = 2.0 * random_vector(local_rand_state) - Vector3f(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

PBRT_GPU Vector3f reflect(const Vector3f &v, const Vector3f &n) {
    return v - 2.0f * v.dot(n) * n;
}

class Material {
  public:
    PBRT_GPU virtual ~Material() {}

    PBRT_GPU virtual bool scatter(const Ray &r_in, const Intersection &intersection,
                                  Color &attenuation, Ray &scattered,
                                  curandState *local_rand_state) const = 0;
};

class Lambertian : public Material {
  public:
    ~Lambertian() override = default;

    PBRT_GPU explicit Lambertian(const Color &a) : albedo(a) {}
    PBRT_GPU bool scatter(const Ray &r_in, const Intersection &intersection, Color &attenuation,
                          Ray &scattered, curandState *local_rand_state) const override {
        Point3f target = intersection.p + intersection.n + random_in_unit_sphere(local_rand_state);
        scattered = Ray(intersection.p, target - intersection.p);
        attenuation = albedo;
        return true;
    }

    Color albedo;
};

class Metal : public Material {
  public:
    ~Metal() override = default;

    PBRT_GPU Metal(const Color &a, double f) : albedo(a), fuzz(gpu_clamp_0_1(f)) {}

    PBRT_GPU bool scatter(const Ray &r_in, const Intersection &intersection, Color &attenuation,
                          Ray &scattered, curandState *local_rand_state) const override {
        Vector3f reflected = reflect(r_in.d.normalize(), intersection.n);
        scattered = Ray(intersection.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return scattered.d.dot(intersection.n) > 0.0;
    }
    Color albedo;
    double fuzz;
};

class Dielectric : public Material {
  public:
    ~Dielectric() override = default;

    PBRT_GPU explicit Dielectric(double ri) : ref_idx(ri) {}

    PBRT_GPU bool scatter(const Ray &r_in, const Intersection &intersection, Color &attenuation,
                          Ray &scattered, curandState *local_rand_state) const override {
        Vector3f outward_normal;
        Vector3f reflected = reflect(r_in.d, intersection.n);
        double ni_over_nt;
        attenuation = Color(1.0, 1.0, 1.0);
        Vector3f refracted;
        double reflect_prob;
        double cosine;
        if (r_in.d.dot(intersection.n) > 0.0) {
            outward_normal = -intersection.n;
            ni_over_nt = ref_idx;
            cosine = r_in.d.dot(intersection.n) / r_in.d.length();
            cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
        } else {
            outward_normal = intersection.n;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -r_in.d.dot(intersection.n) / r_in.d.length();
        }

        if (refract(r_in.d, outward_normal, ni_over_nt, refracted)) {
            reflect_prob = schlick(cosine, ref_idx);
        } else {
            reflect_prob = 1.0f;
        }

        if (curand_uniform(local_rand_state) < reflect_prob) {
            scattered = Ray(intersection.p, reflected);
        } else {
            scattered = Ray(intersection.p, refracted);
        }
        return true;
    }

    double ref_idx;
};
