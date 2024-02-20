#pragma once
#include <ray.hpp>
#include <vec3.hpp>
#include <hittable_object.hpp>

class Material {
    public: 
        virtual ~Material() = default;
        virtual bool scatter(const Ray& inRay, const HitRecord& rec, color& attenuation, Ray& scattered) const = 0;
};