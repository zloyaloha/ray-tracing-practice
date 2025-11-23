#pragma once

#include "interval.hpp"
#include "ray.hpp"
#include "vec3.hpp"

const double delta = 1e-2;

class AABB {
private:
    Interval x_inter;
    Interval y_inter;
    Interval z_inter;

public:
    AABB() : x_inter(0, 0), y_inter(0, 0), z_inter(0, 0) {};
    AABB(const Interval& x, const Interval& y, const Interval& z)
        : x_inter(x), y_inter(y), z_inter(z) {
        expand_to_min();
    }
    Interval getX() const { return x_inter; }
    Interval getY() const { return y_inter; }
    Interval getZ() const { return z_inter; }
    AABB(const point3& p1, const point3& p2) {
        x_inter = (p1.x() <= p2.x()) ? Interval(p1.x(), p2.x())
                                     : Interval(p2.x(), p1.x());
        y_inter = (p1.y() <= p2.y()) ? Interval(p1.y(), p2.y())
                                     : Interval(p2.y(), p1.y());
        z_inter = (p1.z() <= p2.z()) ? Interval(p1.z(), p2.z())
                                     : Interval(p2.z(), p1.z());
        expand_to_min();
    }

    void expand_to_min() {
        x_inter = x_inter.len() < delta ? x_inter.expand(delta) : x_inter;
        y_inter = y_inter.len() < delta ? y_inter.expand(delta) : y_inter;
        z_inter = z_inter.len() < delta ? z_inter.expand(delta) : z_inter;
    }

    AABB(const AABB& a, const AABB& b)
        : x_inter(Interval(a.x_inter, b.x_inter)),
          y_inter(Interval(a.y_inter, b.y_inter)),
          z_inter(Interval(a.z_inter, b.z_inter)) {
        expand_to_min();
    }

    bool hit(const Ray& r, Interval ray_t) const {
        const point3& origin = r.origin();
        const vec3& direction = r.direction();

        {
            double invD = 1.0 / direction.x();
            double t0 = (x_inter.min - origin.x()) * invD;
            double t1 = (x_inter.max - origin.x()) * invD;
            if (invD < 0.0) std::swap(t0, t1);

            ray_t.min = t0 > ray_t.min ? t0 : ray_t.min;
            ray_t.max = t1 < ray_t.max ? t1 : ray_t.max;
            if (ray_t.max <= ray_t.min) return false;
        }

        {
            double invD = 1.0 / direction.y();
            double t0 = (y_inter.min - origin.y()) * invD;
            double t1 = (y_inter.max - origin.y()) * invD;
            if (invD < 0.0) std::swap(t0, t1);

            ray_t.min = t0 > ray_t.min ? t0 : ray_t.min;
            ray_t.max = t1 < ray_t.max ? t1 : ray_t.max;
            if (ray_t.max <= ray_t.min) return false;
        }

        {
            double invD = 1.0 / direction.z();
            double t0 = (z_inter.min - origin.z()) * invD;
            double t1 = (z_inter.max - origin.z()) * invD;
            if (invD < 0.0) std::swap(t0, t1);

            ray_t.min = t0 > ray_t.min ? t0 : ray_t.min;
            ray_t.max = t1 < ray_t.max ? t1 : ray_t.max;
            if (ray_t.max <= ray_t.min) return false;
        }

        return true;
    }

    int longest_axis() const {
        double x_len = x_inter.len();
        double y_len = y_inter.len();
        double z_len = z_inter.len();

        if (x_len >= y_len && x_len >= z_len) {
            return 0;
        } else if (y_len >= x_len && y_len >= z_len) {
            return 1;
        } else {
            return 2;
        }
    }
};