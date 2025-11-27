#pragma once

#include "ray.cuh"
#include "vec3.cuh"

class MaterialData;

struct HitRecord {
    point3 point;
    vec3 normal;
    float t;
    bool front_face;
    int material_idx;
    float u;
    float v;

    inline __device__ void set_face_normal(const Ray &r, const vec3 &outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};