#pragma once
#include <material.hpp>

class Lambertian : public Material {
    private:
        color _albedo;
    public:
        Lambertian(const color &a);
        bool scatter(const Ray& inRay, const HitRecord& rec, color& attenuation, Ray& scattered) const override;
};