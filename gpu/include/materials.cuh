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
    float fuzz;
    float ir;
    color absorption;
    color albedo;
    color emit = color(0, 0, 0);
    cudaTextureObject_t tex_obj = 0;
};

inline __device__ static float reflectance(float cosine, float ref_idx) {
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * powf((1 - cosine), 5);
}

inline __device__ bool material_scatter(const Ray &r_in, const HitRecord &rec, color &attenuation, Ray &scattered,
                                        unsigned int &seed, const MaterialData &mat) {
    color base_color = mat.albedo;

    if (mat.tex_obj != 0) {
        float4 tex_val = tex2D<float4>(mat.tex_obj, rec.u, rec.v);
        base_color = color(tex_val.x, tex_val.y, tex_val.z) * mat.albedo;
    }

    switch (mat.type) {
        case LAMBERTIAN: {
            vec3 scatter_direction = random_in_hemisphere(rec.normal, seed);
            if (scatter_direction.near_zero()) scatter_direction = rec.normal;
            scattered = Ray(rec.point, scatter_direction);
            attenuation = base_color;
            return true;
        }

        case METAL: {
            float p_metal = 0.8f;
            if (random_float(seed) < p_metal) {
                vec3 reflected = unit_vector(r_in.direction()).reflect(rec.normal);
                scattered = Ray(rec.point, reflected + mat.fuzz * random_in_unit_sphere(seed));
                attenuation = base_color;
                return dot(scattered.direction(), rec.normal) > 0;
            } else {
                vec3 scatter_direction = random_in_hemisphere(rec.normal, seed);
                if (scatter_direction.near_zero()) scatter_direction = rec.normal;
                scattered = Ray(rec.point, scatter_direction);
                attenuation = base_color;
                return true;
            }
        }

        case DIELECTRIC: {
            attenuation = color(1.0, 1.0, 1.0);
            float refraction_ratio = rec.front_face ? (1.0 / mat.ir) : mat.ir;

            vec3 unit_direction = unit_vector(r_in.direction());
            float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0);
            float sin_theta = sqrtf(1.0 - cos_theta * cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0;
            vec3 direction;

            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(seed)) {
                direction = unit_direction.reflect(rec.normal);
            } else {
                direction = unit_direction.refract(rec.normal, refraction_ratio);
            }

            float distance = (rec.point - r_in.origin()).len();
            color absorbance = mat.absorption;
            color transmission =
                color(exp(-absorbance.x() * distance), exp(-absorbance.y() * distance), exp(-absorbance.z() * distance));

            if (!rec.front_face) {
                attenuation = attenuation * transmission;
            }

            float p = fmaxf(attenuation.x(), fmaxf(attenuation.y(), attenuation.z()));
            if (random_float(seed) > p) return false;
            attenuation /= p;

            float offset = 1e-4f;
            vec3 origin = rec.point + rec.normal * offset * (dot(direction, rec.normal) > 0 ? 1.0f : -1.0f);

            scattered = Ray(origin, direction);

            return true;
        }

        case DIFFUSE_LIGHT: {
            return false;
        }
    }
    return false;
}

inline __device__ color material_emit(const MaterialData &mat) {
    // switch (mat.type) {
    //     case LAMBERTIAN:
    //     case METAL:
    //     case DIELECTRIC: {
    //         return color(0.0, 0.0, 0.0);
    //     }
    //     case DIFFUSE_LIGHT: {
            return mat.emit;
        // }
    // }
    // return color(0.0, 0.0, 0.0);
}