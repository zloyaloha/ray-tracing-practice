#include <lambertian.hpp>

Lambertian::Lambertian(const color& a) : _albedo(a) {}

bool Lambertian::scatter(const Ray& inRay, const HitRecord& rec,
                         color& attenuation, Ray& scattered) const {
    auto scatter_direction = rec.normal + vec3::randomInUnitSphere();
    if (scatter_direction.near_zero()) {
        scatter_direction = rec.normal;
    }
    scattered = Ray(rec.point + rec.normal * 1e-5, scatter_direction);
    attenuation = _albedo;
    return true;
}