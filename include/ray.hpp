#pragma once 
#include <iostream>
#include "vec3.hpp"

using point3 = vec3;

class Ray {
    public:
        Ray() = default;
        Ray(const point3 &point, const vec3 &direction) : basePoint(point), dir(direction) {}
        point3 origin() const { return basePoint; }
        vec3 direction() const { return dir; }
        point3 at(double t) const {
            return basePoint + t*dir;
        }
    private:
        point3 basePoint;
        vec3 dir;
};