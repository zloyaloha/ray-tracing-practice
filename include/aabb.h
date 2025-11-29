#pragma once

#include "interval.h"
#include "ray.h"
#include "vec3.h"

struct AABB {
    Interval x_interval, y_interval, z_interval;

    __host__ __device__ AABB() {}

    __host__ __device__ AABB(const Interval& x, const Interval& y, const Interval& z)
        : x_interval(x), y_interval(y), z_interval(z) {
        expand_to_min();
    }

    __host__ __device__ Interval getX() const { return x_interval; }
    __host__ __device__ Interval getY() const { return y_interval; }
    __host__ __device__ Interval getZ() const { return z_interval; }

    __host__ __device__ AABB(const point3& p1, const point3& p2) {
        x_interval = (p1.x() <= p2.x()) ? Interval(p1.x(), p2.x()) : Interval(p2.x(), p1.x());
        y_interval = (p1.y() <= p2.y()) ? Interval(p1.y(), p2.y()) : Interval(p2.y(), p1.y());
        z_interval = (p1.z() <= p2.z()) ? Interval(p1.z(), p2.z()) : Interval(p2.z(), p1.z());
        expand_to_min();
    }

    __host__ __device__ AABB(const AABB& a, const AABB& b)
        : x_interval(Interval(a.x_interval, b.x_interval)),
          y_interval(Interval(a.y_interval, b.y_interval)),
          z_interval(Interval(a.z_interval, b.z_interval)) {
        expand_to_min();
    }

    __host__ __device__ void pad() {
        double delta = 0.0001;
        if (x_interval.size() < delta) x_interval = x_interval.expand(delta);
        if (y_interval.size() < delta) y_interval = y_interval.expand(delta);
        if (z_interval.size() < delta) z_interval = z_interval.expand(delta);
    }

    __host__ __device__ bool hit(const Ray& r, Interval ray_t) const {
        for (int a = 0; a < 3; a++) {
            auto invD = 1 / r.direction()[a];
            auto orig = r.origin()[a];

            float min_val = (a == 0) ? x_interval.min : ((a == 1) ? y_interval.min : z_interval.min);
            float max_val = (a == 0) ? x_interval.max : ((a == 1) ? y_interval.max : z_interval.max);

            auto t1 = (min_val - orig) * invD;
            auto t2 = (max_val - orig) * invD;

            if (invD < 0) {
                float temp = t1;
                t1 = t2;
                t2 = temp;
            }

            if (t1 > ray_t.min) ray_t.min = t1;
            if (t2 < ray_t.max) ray_t.max = t2;

            if (ray_t.max <= ray_t.min) return false;
        }
        return true;
    }

    __host__ __device__ bool hit(const Ray& r, const vec3& invD, Interval ray_t) const {
        for (int a = 0; a < 3; a++) {
            auto orig = r.origin()[a];
            auto invD_val = invD[a];

            float min_val = (a == 0) ? x_interval.min : ((a == 1) ? y_interval.min : z_interval.min);
            float max_val = (a == 0) ? x_interval.max : ((a == 1) ? y_interval.max : z_interval.max);

            auto t1 = (min_val - orig) * invD_val;
            auto t2 = (max_val - orig) * invD_val;

            if (invD_val < 0) {
                float temp = t1;
                t1 = t2;
                t2 = temp;
            }

            if (t1 > ray_t.min) ray_t.min = t1;
            if (t2 < ray_t.max) ray_t.max = t2;

            if (ray_t.max <= ray_t.min) return false;
        }
        return true;
    }

    __host__ __device__ void expand_to_min() {
        double delta = 0.0001;
        x_interval = x_interval.size() < delta ? x_interval.expand(delta) : x_interval;
        y_interval = y_interval.size() < delta ? y_interval.expand(delta) : y_interval;
        z_interval = z_interval.size() < delta ? z_interval.expand(delta) : z_interval;
    }
};

__host__ __device__ inline AABB surround(const AABB& box0, const AABB& box1) { return AABB(box0, box1); }