#pragma once

struct Intersection;

#include "base/ray.h"
#include "base/shape.h"

__device__ double schlick(double cosine, double ref_idx) {
    double r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const Vector3 &v, const Vector3 &n, double ni_over_nt, Vector3 &refracted) {
    Vector3 uv = v.normalize();
    double dt = dot(uv, n);
    double discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);

    if (discriminant <= 0) {
        return false;
    }

    refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
    return true;
}

__device__ Vector3 random_vector(curandState *local_rand_state) {
    return Vector3(curand_uniform(local_rand_state), curand_uniform(local_rand_state),
                   curand_uniform(local_rand_state));
}

__device__ Vector3 random_in_unit_sphere(curandState *local_rand_state) {
    Vector3 p;
    do {
        p = 2.0f * random_vector(local_rand_state) - Vector3(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ Vector3 reflect(const Vector3 &v, const Vector3 &n) {
    return v - 2.0f * dot(v, n) * n;
}

class Material {
    public:
        __device__ virtual ~Material() {}

        __device__ virtual bool scatter(const Ray &r_in, const Intersection &intersection, Color &attenuation,
                                        Ray &scattered, curandState *local_rand_state) const = 0;
};

class Lambertian : public Material {
    public:
        ~Lambertian() override = default;

        __device__ explicit Lambertian(const Color &a) : albedo(a) {}
        __device__ bool scatter(const Ray &r_in, const Intersection &intersection, Color &attenuation,
                                Ray &scattered, curandState *local_rand_state) const override {
            Point target = intersection.p + intersection.n + random_in_unit_sphere(local_rand_state);
            scattered = Ray(intersection.p, target - intersection.p);
            attenuation = albedo;
            return true;
        }

        Color albedo;
};

class Metal : public Material {
    public:
        ~Metal() override = default;

        __device__ Metal(const Color &a, double f) : albedo(a), fuzz(gpu_clamp_0_1(f)) {}

        __device__ bool scatter(const Ray &r_in, const Intersection &intersection, Color &attenuation,
                                Ray &scattered, curandState *local_rand_state) const override {
            Vector3 reflected = reflect(r_in.d.normalize(), intersection.n);
            scattered = Ray(intersection.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            return (dot(scattered.d, intersection.n) > 0.0f);
        }
        Color albedo;
        double fuzz;
};

class Dielectric : public Material {
    public:
        ~Dielectric() override = default;

        __device__ explicit Dielectric(double ri) : ref_idx(ri) {}

        __device__ bool scatter(const Ray &r_in, const Intersection &intersection, Color &attenuation,
                                Ray &scattered, curandState *local_rand_state) const override {
            Vector3 outward_normal;
            Vector3 reflected = reflect(r_in.d, intersection.n);
            double ni_over_nt;
            attenuation = Color(1.0, 1.0, 1.0);
            Vector3 refracted;
            double reflect_prob;
            double cosine;
            if (dot(r_in.d, intersection.n) > 0.0f) {
                outward_normal = -intersection.n;
                ni_over_nt = ref_idx;
                cosine = dot(r_in.d, intersection.n) / r_in.d.length();
                cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
            } else {
                outward_normal = intersection.n;
                ni_over_nt = 1.0f / ref_idx;
                cosine = -dot(r_in.d, intersection.n) / r_in.d.length();
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
