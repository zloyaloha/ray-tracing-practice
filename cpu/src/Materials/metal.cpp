#include "metal.hpp"

Metal::Metal(const color &a, const double &fuzz) : _albedo(a), _blur(fuzz) {}

bool Metal::scatter(const Ray& inRay, const HitRecord& rec, color& attenuation, Ray& reflected) const {
    auto reflected_dir = inRay.direction().reflect(rec.normal);
    reflected = Ray(rec.point, _blur * vec3::randomInUnitSphere() + reflected_dir);
    attenuation = _albedo;
    return (dot(reflected.direction(), rec.normal) > 0);

}