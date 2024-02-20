#pragma once 

#include <cmath>
#include <iostream>
#include <array>
#include <different.hpp>

class vec3 {
    private:
        std::array<double, 3> ref;
    public:
        vec3() : ref{0, 0, 0} {};
        vec3(double e0, double e1, double e2) : ref{e0, e1, e2} {}
        double x() const { return ref[0]; }
        double y() const { return ref[1]; }
        double z() const { return ref[2]; }
        vec3 operator-() const { return vec3(-ref[0], -ref[1], -ref[2]); }

        double operator[](size_t i) const;
        double& operator[](size_t i);

        vec3& operator+=(const vec3 &other);
        vec3& operator*=(const int &alpha);
        vec3& operator/=(const int &alpha);

        double len_squared() const;
        double len() const;

        friend std::ostream& operator<<(std::ostream &os, const vec3 &v);
        friend vec3 operator+(const vec3 &v1, const vec3 &v2);
        friend vec3 operator-(const vec3 &v1, const vec3 &v2);
        friend vec3 operator+(const vec3 &v1, const vec3 &v2);
        friend vec3 operator*(const vec3 &v1, const vec3 &v2);
        friend vec3 operator*(double t, const vec3 &v2);
        friend vec3 operator*(const vec3 &v2, double t);
        friend vec3 operator/(const vec3 &v2, double t);

        friend double dot(const vec3 &v1, const vec3 &v2); 
        friend vec3 cross(const vec3 &v1, const vec3 &v2);
        vec3 unit_vector() const;

        static vec3 randomInUnitSphere();
        static vec3 randomUnitVectorInSphere();
        static vec3 randomUnitVectorInHemisphere(const vec3 &normal);
        static vec3 random();
        static vec3 random(const double &min, const double &max);
};

using point3 = vec3;
using color = vec3;