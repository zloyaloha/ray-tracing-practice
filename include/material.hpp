#pragma once
#include <ray.hpp>
#include <hittable_object.hpp>

class HitRecord;

class Material {
    public: 
        virtual ~Material() = default;
        virtual bool scatter(const Ray& inRay, const HitRecord& rec, color& attenuation, Ray& scattered) const = 0;
};
