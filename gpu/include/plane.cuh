#pragma once

#include <memory>

#include "hittable_object.cuh"

enum PlaneType { QUAD, ELLIPSE, TRIANGLE };

struct __align__(16) PlaneData {
    PlaneType type;
    float D;
    int material_idx;
    vec3 w, u, v;
    point3 base;
    vec3 normal;
    __host__ PlaneData::PlaneData(const point3 &base, const vec3 &u, const vec3 &v, int material_index, PlaneType type);
};

__host__ inline PlaneData::PlaneData(const point3 &base, const vec3 &u, const vec3 &v, int material_index, PlaneType type)
    : material_idx(material_index), base(base), u(u), v(v), type(type) {
    auto n = cross(u, v);

    normal = unit_vector(n);

    D = dot(normal, base);

    w = n / dot(n, n);
}

inline __device__ bool is_interior_quad(double a, double b, HitRecord &rec) {
    Interval unit_interval = Interval(0, 1);

    if (!unit_interval.contains(a) || !unit_interval.contains(b)) return false;

    rec.u = a;
    rec.v = b;
    return true;
}

inline __device__ bool is_interior_ellipse(double a, double b, HitRecord &rec) {
    if (powf(a - 0.5, 2) + powf(b - 0.5, 2) > 0.25) return false;

    rec.u = a;
    rec.v = b;
    return true;
}

inline __device__ bool is_interior_triangle(double a, double b, HitRecord &rec) {
    if (a < 0 || b < 0 || (a + b) > 1) {
        return false;
    }
    rec.u = a;
    rec.v = b;
    return true;
}

inline __device__ bool hit_plane(const Ray &r, const Interval &ray_inter, HitRecord &rec, const PlaneData &plane_data) {
    double denom = dot(plane_data.normal, r.direction());
    if (fabsf(denom) < 1e-8) {
        return false;
    }
    auto root = (plane_data.D - dot(plane_data.normal, r.origin())) / denom;
    if (!ray_inter.contains(root)) {
        return false;
    }

    auto intersection = r.at(root);
    vec3 planar_hitpt_vector = intersection - plane_data.base;
    auto alpha = dot(plane_data.w, cross(planar_hitpt_vector, plane_data.v));
    auto beta = dot(plane_data.w, cross(plane_data.u, planar_hitpt_vector));

    switch (plane_data.type) {
        case (QUAD): {
            if (!is_interior_quad(alpha, beta, rec)) return false;
            break;
        }
        case (ELLIPSE): {
            if (!is_interior_ellipse(alpha, beta, rec)) return false;
            break;
        }
        case (TRIANGLE): {
            if (!is_interior_triangle(alpha, beta, rec)) return false;
            break;
        }
        default:
            break;
    }

    rec.t = root;
    rec.point = r.at(root);
    vec3 vector = rec.point - plane_data.base;
    rec.set_face_normal(r, plane_data.normal);
    rec.material_idx = plane_data.material_idx;

    return true;
}