#pragma once
#include <interval.hpp>
#include <material.hpp>
#include <memory>
#include <ray.hpp>
#include <vec3.hpp>

#include "aabb.hpp"

class Material;

struct HitRecord {
    point3 point;
    vec3 normal;
    double param;
    std::shared_ptr<Material> material;
    bool orientation;
    double u;
    double v;

    HitRecord() = default;
    HitRecord(const point3 &p, const vec3 &n, const double &t);

    void set_orientation(const Ray &ray, const vec3 &outside_normal);
};

class HittableObject {
public:
    virtual ~HittableObject() = default;
    virtual bool hit(const Ray &r, const Interval &ray_inter,
                     HitRecord &rec) const = 0;
    virtual AABB bounding_box() const = 0;
};
