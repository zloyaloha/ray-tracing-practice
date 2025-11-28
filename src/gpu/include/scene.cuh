#pragma once

#include "hittable_object.cuh"
#include "plane.cuh"
#include "random_utils.cuh"
#include "sphere.cuh"

#include "bvh.cuh"

struct SceneData {
    SphereData *d_spheres;
    int num_spheres;

    PlaneData *d_planes;
    int num_planes;

    MaterialData *d_materials;
    int num_materials;

    BVHTree *d_bvh_trees;
    int num_bvh_trees;
};

inline __host__ __device__ bool hit_scene(const Ray &r, Interval ray_t, HitRecord &rec, const SceneData& scene_data_ref) {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_t.max;

    for (int k = 0; k < scene_data_ref.num_bvh_trees; ++k) {
        if (hit_bvh(r, Interval(ray_t.min, closest_so_far), temp_rec, scene_data_ref.d_bvh_trees[k].nodes, scene_data_ref.d_bvh_trees[k].num_nodes, scene_data_ref.d_spheres, scene_data_ref.d_planes)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    for (int k = 0; k < scene_data_ref.num_spheres; ++k) {
        if (hit_sphere(r, Interval(ray_t.min, closest_so_far), temp_rec, scene_data_ref.d_spheres[k])) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    for (int k = 0; k < scene_data_ref.num_planes; ++k) {
        if (hit_plane(r, Interval(1e-2f, closest_so_far), temp_rec, scene_data_ref.d_planes[k])) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}