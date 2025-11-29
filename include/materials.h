#pragma once

#include <curand_kernel.h>

#include <cmath>

#include "hittable_object.h"
#include "random_utils.h"
#include "ray.h"
#include "vec3.h"

enum MaterialType { LAMBERTIAN, METAL, DIELECTRIC, DIFFUSE_LIGHT };

struct CpuTexture {
    float *data;
    int width;
    int height;
};

inline __host__ color tex2D_cpu(CpuTexture *tex, float u, float v) {
    if (!tex || !tex->data) return color(1, 1, 1);

    u = u - floor(u);
    v = v - floor(v);

    float px = u * tex->width;
    float py = (1.0f - v) * tex->height;

    int x0 = static_cast<int>(px);
    int y0 = static_cast<int>(py);
    int x1 = (x0 + 1) % tex->width;
    int y1 = (y0 + 1) % tex->height;

    float dx = px - x0;
    float dy = py - y0;

    auto get_pixel = [&](int x, int y) {
        int idx = (y * tex->width + x) * 4;  // 4 channels
        return color(tex->data[idx], tex->data[idx + 1], tex->data[idx + 2]);
    };

    color c00 = get_pixel(x0, y0);
    color c10 = get_pixel(x1, y0);
    color c01 = get_pixel(x0, y1);
    color c11 = get_pixel(x1, y1);

    color top = c00 * (1.0f - dx) + c10 * dx;
    color bot = c01 * (1.0f - dx) + c11 * dx;

    return top * (1.0f - dy) + bot * dy;
}

struct MaterialData {
    MaterialType type;
    float fuzz;
    float ir;
    color absorption;
    color albedo;
    color emit = color(0, 0, 0);
    cudaTextureObject_t tex_obj = 0;
    CpuTexture *cpu_tex = nullptr;
};

inline __host__ __device__ static float reflectance(float cosine, float ref_idx) {
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * powf((1 - cosine), 5);
}

inline __host__ __device__ bool material_scatter(const Ray &r_in, const HitRecord &rec, color &attenuation, Ray &scattered,
                                                 unsigned int &seed, const MaterialData &mat) {
    switch (mat.type) {
        case LAMBERTIAN: {
            vec3 scatter_direction = random_in_hemisphere(rec.normal, seed);
            if (scatter_direction.near_zero()) scatter_direction = rec.normal;
            scattered = Ray(rec.point, scatter_direction);
            attenuation = mat.albedo;
            return true;
        }

        case METAL: {
            float p_metal = 0.8f;
            if (random_float(seed) < p_metal) {
                vec3 reflected = unit_vector(r_in.direction()).reflect(rec.normal);
                scattered = Ray(rec.point, reflected + mat.fuzz * random_in_unit_sphere(seed));
                attenuation = mat.albedo;
                return dot(scattered.direction(), rec.normal) > 0;
            } else {
                vec3 scatter_direction = random_in_hemisphere(rec.normal, seed);
                if (scatter_direction.near_zero()) scatter_direction = rec.normal;
                scattered = Ray(rec.point, scatter_direction);
                attenuation = mat.albedo;
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

inline __host__ __device__ color material_emit(const MaterialData &mat) { return mat.emit; }