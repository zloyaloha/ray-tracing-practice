#include <ray.hpp>

Ray::Ray(const point3 &point, const vec3 &direction) : basePoint(point), dir(direction) {}

point3 Ray::origin() const { return basePoint; }

vec3 Ray::direction() const { return dir; }

point3 Ray::at(double t) const {
    return basePoint + t*dir;
}