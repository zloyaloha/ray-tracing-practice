#pragma once

#include "hittable_object.cuh"
#include "interval.cuh"
#include "ray.cuh"
#include "vec3.cuh"

struct __align__(16) SphereData {
    point3 center;
    float radius;
    int material_idx;
    __host__ __device__ SphereData(const point3 &center_sphere, float radius_sphere, int material_idx_sphere)
        : center(center_sphere), radius(radius_sphere), material_idx(material_idx_sphere) {}
};

inline __device__ void get_sphere_uv(const vec3 &p, float &u, float &v) {
    float theta = acosf(p.y());
    float phi = atan2f(-p.z(), p.x()) + M_PI;

    u = phi / (2 * M_PI);
    v = theta / M_PI;
}

inline __device__ bool hit_sphere(const Ray &r, const Interval &ray_inter, HitRecord &rec, const SphereData &sphere_data) {
    vec3 oc = r.origin() - sphere_data.center;
    auto a = r.direction().len_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.len_squared() - sphere_data.radius * sphere_data.radius;
    auto D = half_b * half_b - a * c;

    if (D < 0) {
        return false;
    }
    double sqrtD = sqrtf(D);
    double root = (-half_b - sqrtD) / a;

    if (!ray_inter.contains(root)) {
        root = (-half_b + sqrtD) / a;
        if (!ray_inter.contains(root)) {
            return false;
        }
    }

    rec.t = root;
    rec.point = r.at(rec.t);
    vec3 outward_normal = (rec.point - sphere_data.center) / sphere_data.radius;
    rec.set_face_normal(r, outward_normal);
    rec.material_idx = sphere_data.material_idx;
    get_sphere_uv(outward_normal, rec.u, rec.v);

    return true;
}
