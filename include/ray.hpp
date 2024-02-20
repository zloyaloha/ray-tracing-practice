#pragma once 

#include <iostream>
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