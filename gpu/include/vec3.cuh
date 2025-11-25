#pragma once

#include <cmath>
#include <cstddef>

struct vec3;

__host__ __device__ vec3 operator+(const vec3 &v1, const vec3 &v2);
__host__ __device__ vec3 operator-(const vec3 &v1, const vec3 &v2);
__host__ __device__ vec3 operator*(const vec3 &v1, const vec3 &v2);
__host__ __device__ vec3 operator*(float t, const vec3 &v);
__host__ __device__ vec3 operator*(const vec3 &v, float t);
__host__ __device__ vec3 operator/(const vec3 &v, float t);
__host__ __device__ float dot(const vec3 &v1, const vec3 &v2);
__host__ __device__ vec3 cross(const vec3 &v1, const vec3 &v2);
__host__ __device__ vec3 unit_vector(const vec3 &v);

struct vec3 {
    float e[3];

    __host__ __device__ vec3() = default;
    __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    __host__ __device__ float x() const { return e[0]; }
    __host__ __device__ float y() const { return e[1]; }
    __host__ __device__ float z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

    __host__ __device__ float operator[](std::size_t i) const { return e[i]; }
    __host__ __device__ float &operator[](std::size_t i) { return e[i]; }

    __host__ __device__ vec3 &operator+=(const vec3 &other) {
        e[0] += other[0];
        e[1] += other[1];
        e[2] += other[2];
        return *this;
    }

    __host__ __device__ vec3 &operator*=(float alpha) {
        e[0] *= alpha;
        e[1] *= alpha;
        e[2] *= alpha;
        return *this;
    }

    __host__ __device__ vec3 &operator/=(float alpha) { return *this *= (1.0 / alpha); }

    __host__ __device__ float len_squared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
    __host__ __device__ float len() const { return sqrt(len_squared()); }

    __host__ __device__ bool near_zero() const {
        const float s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }

    __host__ __device__ vec3 reflect(const vec3 &n) const { return *this - 2 * dot(*this, n) * n; }

    __device__ vec3 refract(const vec3 &n, float etai_over_etat) const {
        float cos_theta = fmin(dot(-(*this), n), 1.0);
        vec3 r_out_perp = etai_over_etat * ((*this) + cos_theta * n);
        vec3 r_out_parallel = -__fsqrt_rn(fabs(1.0 - r_out_perp.len_squared())) * n;
        return r_out_perp + r_out_parallel;
    }
};

using point3 = vec3;
using color = vec3;

__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
    return vec3(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
    return vec3(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
    return vec3(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) { return vec3(t * v[0], t * v[1], t * v[2]); }

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) { return t * v; }

__host__ __device__ inline vec3 operator/(const vec3 &v, float t) { return (1.0 / t) * v; }

__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2) { return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]; }

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    return vec3(v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3 &v) { return v / v.len(); }