#pragma once
#include "material.hpp"

class Dielectric : public Material {
    private:
        double ir;
        static double reflectance(double cosinus, double refCoef);
    public:
        Dielectric(double sneilsCoef);
        bool scatter(const Ray& inRay, const HitRecord& rec, color& attenuation, Ray& scattered) const override;
};

