#pragma once
#include "vec3.hpp"

class Ray {
public:
    Ray() = default;
    Ray(const point3 &point, const vec3 &direction);
    point3 origin() const;
    vec3 direction() const;
    point3 at(double t) const;

private:
    point3 basePoint;
    vec3 dir;
};

inline Ray::Ray(const point3 &point, const vec3 &direction)
    : basePoint(point), dir(direction) {}

inline point3 Ray::origin() const { return basePoint; }

inline vec3 Ray::direction() const { return dir; }

inline point3 Ray::at(double t) const { return basePoint + t * dir; }