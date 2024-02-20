#pragma once
#include "hittable_object.hpp"
#include "interval.hpp"
#include "ray.hpp"
#include "vec3.hpp"
#include <memory>

class Sphere : public HittableObject {
    public:
        Sphere(const point3 &center, const double &raduis, std::shared_ptr<Material> material);
        bool hit(const Ray& r, const Interval &ray_inter, HitRecord& rec) const override;
    private: 
        point3 _center;
        double _radius;
        std::shared_ptr<Material> _material;
};