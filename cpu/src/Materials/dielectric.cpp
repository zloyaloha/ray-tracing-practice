#include <dielectric.hpp>

Dielectric::Dielectric(double sneilsCoef) : ir(sneilsCoef) {}

bool Dielectric::scatter(const Ray& inRay, const HitRecord& rec, color& attenuation, Ray& scattered) const {
    attenuation = color(1.0, 1.0, 1.0);
    double refraction_ratio = rec.orientation ? (1.0 / ir) : ir;
    double cos_theta = fmin(dot(-inRay.direction(), rec.normal), 1.0);
    double sin_theta = std::sqrt(1 - pow(cos_theta, 2));
    bool is_refr = (refraction_ratio * sin_theta <= 1);
    vec3 vector;
    if (!is_refr || reflectance(cos_theta, refraction_ratio) > randomDouble()) {
        vector = inRay.direction().reflect(rec.normal);
    } else { 
        vector = inRay.direction().unit_vector().refract(rec.normal, refraction_ratio);
    }
    scattered = Ray(rec.point, vector);
    return true;
}

double Dielectric::reflectance(double cosinus, double refCoef) {
    double r0 = (1 - refCoef) / (1 + refCoef);
    r0 = pow(r0, 2);
    return r0 + (1 - r0) * pow(1 - cosinus, 5);
}