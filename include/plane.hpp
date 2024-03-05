#pragma once

#include "hittable_object.hpp"

class Plane : public HittableObject {
    private:
        point3 _base;
        vec3 _normal;
        double _radius;
        std::shared_ptr<Material> _material;
    public:
        Plane(const point3 &base, double radius, const vec3 &normal, std::shared_ptr<Material> mat);
        bool hit(const Ray& r, const Interval &ray_inter, HitRecord& rec) const override;
};