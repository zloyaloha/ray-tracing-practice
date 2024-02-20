#include <hittable_object.hpp>

HitRecord::HitRecord(const point3 &p, const vec3 &n, const double &t) : point(p), normal(n), param(t) {}

void HitRecord::set_orientation(const Ray &ray, const vec3 &outside_normal) {
    orientation = dot(ray.direction(), outside_normal) < 0;
    normal = orientation ? outside_normal : -outside_normal;
}