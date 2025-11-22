#include "sphere.hpp"

Sphere::Sphere(const point3 &center, const double &raduis,
               std::shared_ptr<Material> material)
    : _center(center), _radius(raduis), _material(material) {}

bool Sphere::hit(const Ray &r, const Interval &ray_inter,
                 HitRecord &rec) const {
    vec3 oc = r.origin() - _center;
    auto a = r.direction().len_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.len_squared() - _radius * _radius;
    auto D = half_b * half_b - a * c;

    if (D < 0) {
        return false;
    }
    double sqrtD = std::sqrt(D);
    double root = (-half_b - sqrtD) / a;

    if (!ray_inter.contains(root)) {
        root = (-half_b + sqrtD) / a;
        if (!ray_inter.contains(root)) {
            return false;
        }
    }

    rec.param = root;
    rec.point = r.at(rec.param);
    vec3 outward_normal = (rec.point - _center) / _radius;
    rec.set_orientation(r, outward_normal);
    rec.material = _material;

    return true;
}