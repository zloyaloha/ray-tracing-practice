#pragma once
#include <hittable_object.hpp>
#include <ray.hpp>

class HitRecord;

class Material {
public:
    virtual ~Material() = default;
    virtual bool scatter(const Ray& inRay, const HitRecord& rec,
                         color& attenuation, Ray& scattered) const = 0;
};
