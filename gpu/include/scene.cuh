#pragma once

#include "hittable_object.cuh"
#include "sphere.cuh"

struct SceneData {
    SphereData *d_spheres;
    int num_spheres;

    MaterialData *d_materials;
    int num_materials;
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

    return hit_anything;
}