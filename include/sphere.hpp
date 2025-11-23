#pragma once
#include <memory>

#include "aabb.hpp"
#include "hittable_object.hpp"
#include "interval.hpp"
#include "ray.hpp"
#include "vec3.hpp"

class Sphere : public HittableObject {
public:
    Sphere(const point3 &center, const double &raduis,
           std::shared_ptr<Material> material);
    bool hit(const Ray &r, const Interval &ray_inter,
             HitRecord &rec) const override;

    AABB bounding_box() const override;

private:
    point3 _center;
    double _radius;
    std::shared_ptr<Material> _material;
    AABB bbox;
};
