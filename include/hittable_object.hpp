#pragma once
#include <ray.hpp>
#include "interval.hpp"

struct HitRecord {

    enum OrientationType {
        outside = 0,
        inside = 1
    };

    point3 point;
    vec3 normal;
    double param;
    bool orientation;
    HitRecord() = default;

    HitRecord(const point3 &p, const vec3 &n, const double &t) : point(p), normal(n), param(t) {}

    void set_orientation(const Ray &ray, const vec3 &outside_normal) {
        orientation = dot(ray.direction(), outside_normal) > 0;
        normal = orientation ? -outside_normal : outside_normal;
    }

};

class HittableObject {
    public:
        virtual ~HittableObject() = default;

        virtual bool hit(const Ray& r, const Interval &ray_inter, HitRecord& rec) const = 0;
};
