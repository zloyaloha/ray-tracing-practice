#pragma once

#include "hittable_object.cuh"
#include "plane.cuh"
#include "random_utils.cuh"
#include "sphere.cuh"

enum LightType { PLANE_LIGHT, SPHERE_LIGHT };

struct LightData {
    LightType type;
    color emission;
    union {
        struct {
            point3 p0;
            vec3 e1, e2;
            vec3 normal;
        } quad;
        struct {
            point3 center;
            float radius;
        } sphere;
        struct {
            point3 position;
        } point;
        struct {
            vec3 direction;
        } directional;
    };
};

__device__ inline point3 sample(const LightData &L, const HitRecord &rec, float &pdf_light, unsigned int &seed) {
    switch (L.type) {
        case (PLANE_LIGHT): {
            float u = random_float(seed);
            float v = random_float(seed);

            point3 p = L.quad.p0 + u * L.quad.e1 + v * L.quad.e2;

            float area = L.quad.e1.len() * L.quad.e2.len();
            pdf_light = 1.0f / area;

            return p;
        }
        case SPHERE_LIGHT: {
            float theta = 2.0f * M_PI * random_float(seed);       // [0, 2π)
            float phi = acosf(2.0f * random_float(seed) - 1.0f);  // [0, π]

            float sin_phi = sinf(phi);
            vec3 dir(sin_phi * cosf(theta), sin_phi * sinf(theta), cosf(phi));

            point3 p = L.sphere.center + L.sphere.radius * dir;

            float surface_area = 4.0f * M_PI * L.sphere.radius * L.sphere.radius;
            pdf_light = 1.0f / surface_area;

            return p;
        }
        default:
            pdf_light = 0.0f;  // Safety
            return point3(0, 0, 0);
    }
}

__device__ inline color emission(const LightData &L, const point3 &p) { return L.emission; }

struct SceneData {
    SphereData *d_spheres;
    int num_spheres;

    PlaneData *d_planes;
    int num_planes;

    MaterialData *d_materials;
    int num_materials;

    LightData *d_lights;
    int num_lights;
};

extern __constant__ SceneData d_scene_data_const;

inline __device__ bool hit_scene(const Ray &r, Interval ray_t, HitRecord &rec) {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_t.max;

    for (int k = 0; k < d_scene_data_const.num_spheres; ++k) {
        if (hit_sphere(r, Interval(ray_t.min, closest_so_far), temp_rec, d_scene_data_const.d_spheres[k])) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    for (int k = 0; k < d_scene_data_const.num_planes; ++k) {
        if (hit_plane(r, Interval(1e-2f, closest_so_far), temp_rec, d_scene_data_const.d_planes[k])) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    // for (int k = 0; k < d_scene_data_const.num_lights; ++k) {
    //     if (hit_light(r, Interval(ray_t.min, closest_so_far), temp_rec, d_scene_data_const.num_lights[k])) {
    //         hit_anything = true;
    //         closest_so_far = temp_rec.t;
    //         rec = temp_rec;
    //     }
    // }

    return hit_anything;
}

inline __device__ color direct_light(const HitRecord &rec, const MaterialData &mat, unsigned int &seed) {
    color result(0, 0, 0);
    int nlights = d_scene_data_const.num_lights;

    for (int i = 0; i < nlights; i++) {
        LightData L = d_scene_data_const.d_lights[i];

        float pdf_light;                             // PDF w.r.t. Area
        point3 p = sample(L, rec, pdf_light, seed);  // FIX: Передаем seed

        vec3 to_light = p - rec.point;
        float dist = to_light.len();
        float dist2 = dist * dist;
        vec3 wi = to_light / dist;

        if (pdf_light <= 0.0f) continue;  // Safety check

        HitRecord shadow_rec;
        // Проверка тени
        if (hit_scene(Ray(rec.point, wi), Interval(0.001f, dist - 0.001f), shadow_rec)) continue;

        float nl = dot(rec.normal, wi);
        if (nl <= 0) continue;

        float cos_light;
        if (L.type == PLANE_LIGHT) {
            cos_light = fabs(dot(L.quad.normal, -wi));
            if (cos_light <= 0) continue;
        } else {
            cos_light = 1;  // Для точечных/сферических упрощение
        }

        color Le = emission(L, p);
        color brdf = mat.albedo / M_PI;  // Простой Ламберт

        float G = (nl * cos_light) / dist2;
        result += Le * brdf * (G / pdf_light);
    }

    return result;
}
