#pragma once

#include "hittable_object.cuh"
#include "interval.cuh"
#include "ray.cuh"
#include "vec3.cuh"

struct SphereData {
    point3 center;
    float radius;
    int material_idx;
    __host__ __device__ SphereData(const point3 &center_sphere, float radius_sphere, int material_idx_sphere)
        : center(center_sphere), radius(radius_sphere), material_idx(material_idx_sphere) {}
};

inline __device__ bool hit_sphere(const Ray &r, const Interval &ray_t, HitRecord &rec, const SphereData &sphere_data) {
    vec3 oc = r.origin() - sphere_data.center;
    float a = r.direction().len_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.len_squared() - sphere_data.radius * sphere_data.radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0) return false;
    if (a < 1e-8) return false;  // Защита от деления на ноль
    float sqrtd = sqrt(discriminant);

    float root1 = (-half_b - sqrtd) / a;
    float root2 = (-half_b + sqrtd) / a;

    float root = -1.0;
    if (root1 >= ray_t.min && root1 <= ray_t.max) {
        root = root1;
    }
    if (root2 >= ray_t.min && root2 <= ray_t.max) {
        // Если root1 тоже валиден, выбираем ближайший
        if (root < 0 || root2 < root) {
            root = root2;
        }
    }

    if (root < 0) {
        return false;
    }

    rec.t = root;
    rec.point = r.at(rec.t);
    vec3 outward_normal = (rec.point - sphere_data.center) / sphere_data.radius;
    rec.set_face_normal(r, outward_normal);
    rec.material_idx = sphere_data.material_idx;
    return true;
}
