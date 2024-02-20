#pragma once
#include "hittable_object.hpp"
#include "vec3.hpp"

class Sphere : public HittableObject {
    public:
        Sphere(const point3 &center, const double &raduis);
        bool hit(const Ray& r, const Interval &ray_inter, HitRecord& rec) const override;
    private: 
        point3 _center;
        double _radius;
};