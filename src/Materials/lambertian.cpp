#include <lambertian.hpp>

Lambertian::Lambertian(const color &a) : _albedo(a) {}

bool Lambertian::scatter(const Ray& inRay, const HitRecord& rec, color& attenuation, Ray& scattered) const {
    auto scatter_direction = rec.normal + vec3::randomUnitVectorInSphere();
    scattered = Ray(rec.point, scatter_direction);
    attenuation = _albedo;
    return true;
}