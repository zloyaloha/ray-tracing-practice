#pragma once
#include <vec3.hpp>
#include <material.hpp>
#include <interval.hpp>
#include <ray.hpp>

class Material;

struct HitRecord {
    point3 point;
    vec3 normal;
    double param;
    std::shared_ptr<Material> material;
    bool orientation;

    HitRecord() = default;
    HitRecord(const point3 &p, const vec3 &n, const double &t);

    void set_orientation(const Ray &ray, const vec3 &outside_normal);
};

class HittableObject {
    public:
        virtual ~HittableObject() = default;
        virtual bool hit(const Ray& r, const Interval &ray_inter, HitRecord& rec) const = 0;
};
