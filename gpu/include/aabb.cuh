#pragma once

#include "interval.cuh"
#include "ray.cuh"
#include "vec3.cuh"

const float delta = 1e-2;

class AABB {
private:
    Interval x_inter;
    Interval y_inter;
    Interval z_inter;

public:
    __host__ __device__ AABB() : x_inter(0, 0), y_inter(0, 0), z_inter(0, 0) {};
    __host__ __device__ AABB(const Interval& x, const Interval& y, const Interval& z) : x_inter(x), y_inter(y), z_inter(z) {
        expand_to_min();
    }
    __host__ __device__ Interval getX() const { return x_inter; }
    __host__ __device__ Interval getY() const { return y_inter; }
    __host__ __device__ Interval getZ() const { return z_inter; }
    __host__ __device__ AABB(const point3& p1, const point3& p2) {
        x_inter = (p1.x() <= p2.x()) ? Interval(p1.x(), p2.x()) : Interval(p2.x(), p1.x());
        y_inter = (p1.y() <= p2.y()) ? Interval(p1.y(), p2.y()) : Interval(p2.y(), p1.y());
        z_inter = (p1.z() <= p2.z()) ? Interval(p1.z(), p2.z()) : Interval(p2.z(), p1.z());
        expand_to_min();
    }

    __host__ __device__ void expand_to_min() {
        x_inter = x_inter.len() < delta ? x_inter.expand(delta) : x_inter;
        y_inter = y_inter.len() < delta ? y_inter.expand(delta) : y_inter;
        z_inter = z_inter.len() < delta ? z_inter.expand(delta) : z_inter;
    }

    __host__ __device__ AABB(const AABB& a, const AABB& b)
        : x_inter(Interval(a.x_inter, b.x_inter)),
          y_inter(Interval(a.y_inter, b.y_inter)),
          z_inter(Interval(a.z_inter, b.z_inter)) {
        expand_to_min();
    }

    __host__ __device__ bool hit(const Ray& r, Interval ray_t) const {
        const point3& origin = r.origin();
        const vec3& direction = r.direction();

        {
            float invD = 1.0 / direction.x();
            float t0 = (x_inter.min - origin.x()) * invD;
            float t1 = (x_inter.max - origin.x()) * invD;
            if (invD < 0.0) std::swap(t0, t1);

            ray_t.min = t0 > ray_t.min ? t0 : ray_t.min;
            ray_t.max = t1 < ray_t.max ? t1 : ray_t.max;
            if (ray_t.max <= ray_t.min) return false;
        }

        {
            float invD = 1.0 / direction.y();
            float t0 = (y_inter.min - origin.y()) * invD;
            float t1 = (y_inter.max - origin.y()) * invD;
            if (invD < 0.0) std::swap(t0, t1);

            ray_t.min = t0 > ray_t.min ? t0 : ray_t.min;
            ray_t.max = t1 < ray_t.max ? t1 : ray_t.max;
            if (ray_t.max <= ray_t.min) return false;
        }

        {
            float invD = 1.0 / direction.z();
            float t0 = (z_inter.min - origin.z()) * invD;
            float t1 = (z_inter.max - origin.z()) * invD;
            if (invD < 0.0) std::swap(t0, t1);

            ray_t.min = t0 > ray_t.min ? t0 : ray_t.min;
            ray_t.max = t1 < ray_t.max ? t1 : ray_t.max;
            if (ray_t.max <= ray_t.min) return false;
        }

        return true;
    }

    __host__ __device__ int longest_axis() const {
        float x_len = x_inter.len();
        float y_len = y_inter.len();
        float z_len = z_inter.len();

        if (x_len >= y_len && x_len >= z_len) {
            return 0;
        } else if (y_len >= x_len && y_len >= z_len) {
            return 1;
        } else {
            return 2;
        }
    }
};