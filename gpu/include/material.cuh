#pragma once

#include <curand_kernel.h>

#include <cmath>

#include "hittable_object.cuh"
#include "random_utils.cuh"
#include "ray.cuh"
#include "vec3.cuh"

enum MaterialType { LAMBERTIAN, METAL, DIELECTRIC, DIFFUSE_LIGHT };

struct MaterialData {
    MaterialType type;
    color albedo;
    float fuzz;
    float ir;
    color emit;
};

inline __device__ static float reflectance(float cosine, float ref_idx) {
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

inline __device__ bool material_scatter(const Ray &r_in, const HitRecord &rec, color &attenuation, Ray &scattered,
                                        unsigned int &seed, const MaterialData &mat) {
    switch (mat.type) {
        case LAMBERTIAN: {
            vec3 scatter_direction = rec.normal + random_unit_vector(seed);
            if (scatter_direction.near_zero()) scatter_direction = rec.normal;
            scattered = Ray(rec.point, scatter_direction);
            attenuation = mat.albedo;
            return true;
        }

        case METAL: {
            vec3 reflected = unit_vector(r_in.direction()).reflect(rec.normal);
            scattered = Ray(rec.point, reflected + mat.fuzz * random_in_unit_sphere(seed));
            attenuation = mat.albedo;
            return dot(scattered.direction(), rec.normal) > 0;
        }

        case DIELECTRIC: {
            attenuation = color(1.0, 1.0, 1.0);
            float refraction_ratio = rec.front_face ? (1.0 / mat.ir) : mat.ir;

            vec3 unit_direction = unit_vector(r_in.direction());
            float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
            float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0;
            vec3 direction;

            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(seed)) {
                direction = unit_direction.reflect(rec.normal);
            } else {
                direction = unit_direction.refract(rec.normal, refraction_ratio);
            }

            scattered = Ray(rec.point, direction);
            return true;
        }

        case DIFFUSE_LIGHT: {
            return false;
        }
    }
    return false;
}

inline __device__ color material_emit(const MaterialData &mat) {
    switch (mat.type) {
        case LAMBERTIAN:
        case METAL:
        case DIELECTRIC: {
            return color(0.0, 0.0, 0.0);
        }
        case DIFFUSE_LIGHT: {
            return mat.emit;
        }
    }
    return color(0.0, 0.0, 0.0);
}

class Material {
public:
    __device__ virtual ~Material() = default;
    __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, color &attenuation, Ray &scattered,
                                    unsigned int &seed) const = 0;
    __device__ virtual color emitted() const { return color(0.0, 0.0, 0.0); }
};

class Lambertian : public Material {
public:
    __host__ __device__ explicit Lambertian(const color &albedo) : albedo(albedo) {}

    __device__ bool scatter(const Ray &, const HitRecord &rec, color &attenuation, Ray &scattered,
                            unsigned int &seed) const override {
        vec3 scatter_direction = rec.normal + random_unit_vector(seed);
        if (scatter_direction.near_zero()) scatter_direction = rec.normal;
        scattered = Ray(rec.point, scatter_direction);
        attenuation = albedo;
        return true;
    }

private:
    color albedo;
};

class Metal : public Material {
public:
    __host__ __device__ Metal(const color &albedo, float fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    __device__ bool scatter(const Ray &r_in, const HitRecord &rec, color &attenuation, Ray &scattered,
                            unsigned int &seed) const override {
        vec3 reflected = unit_vector(r_in.direction()).reflect(rec.normal);
        scattered = Ray(rec.point, reflected + fuzz * random_in_unit_sphere(seed));
        attenuation = albedo;
        return dot(scattered.direction(), rec.normal) > 0;
    }

private:
    color albedo;
    float fuzz;
};

class Dielectric : public Material {
public:
    __host__ __device__ explicit Dielectric(float index_of_refraction) : ir(index_of_refraction) {}

    __device__ bool scatter(const Ray &r_in, const HitRecord &rec, color &attenuation, Ray &scattered,
                            unsigned int &seed) const override {
        attenuation = color(1.0, 1.0, 1.0);
        float refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(seed)) {
            direction = unit_direction.reflect(rec.normal);
        } else {
            direction = unit_direction.refract(rec.normal, refraction_ratio);
        }

        scattered = Ray(rec.point, direction);
        return true;
    }

private:
    float ir;

    __device__ static float reflectance(float cosine, float ref_idx) {
        float r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

class DiffuseLight : public Material {
public:
    __host__ __device__ explicit DiffuseLight(color emit_color) : emit(emit_color) {}

    __device__ bool scatter(const Ray &, const HitRecord &, color &, Ray &, unsigned int &seed) const override { return false; }
    __device__ color emitted() const override { return emit; }

private:
    color emit;
};
