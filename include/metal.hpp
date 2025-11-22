#pragma once
#include "material.hpp"

class Metal : public Material {
private:
    color _albedo;
    double _blur;

public:
    Metal(const color& a, const double& fuzz);
    bool scatter(const Ray& inRay, const HitRecord& rec, color& attenuation,
                 Ray& scattered) const override;
};
