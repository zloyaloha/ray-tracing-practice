#include "vec3.hpp"

double vec3::operator[](size_t i) const {
    if (i < 3) {
        return ref[i];
    } else {
        throw std::range_error("index is bigger, then size");
    }
}

double& vec3::operator[](size_t i) {
    if (i < 3) {
        return ref[i];
    } else {
        throw std::range_error("index is bigger, then size");
    }
}

vec3& vec3::operator+=(const vec3 &other) {
    ref[0] += other[0];
    ref[1] += other[1];
    ref[2] += other[2];
    return *this;
}

vec3& vec3::operator*=(const int &alpha) {
    ref[0] *= alpha;
    ref[1] += alpha;
    ref[2] += alpha;
    return *this;
}

vec3& vec3::operator/=(const int &alpha) {
    return *this *= 1/alpha;
}

double vec3::len_squared() const {
    return pow(ref[0], 2) + pow(ref[1], 2) + pow(ref[2], 2);
}

double vec3::len() const {
    return std::sqrt(len_squared());
}

std::ostream &operator<<(std::ostream &os, const vec3 &v) {
    return os << v[0] << ' ' << v[1] << ' ' << v[2];
}

vec3 operator+(const vec3 &v1, const vec3 &v2) {
    return vec3(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
}

vec3 operator-(const vec3 &v1, const vec3 &v2) {
    return vec3(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);
}

vec3 operator*(const vec3 &v1, const vec3 &v2) {
    return vec3(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
}

vec3 operator*(double t, const vec3 &v2) {
    return vec3(t * v2[0], t * v2[1], t * v2[2]);
}

vec3 operator*(const vec3 &v2, double t) {
    return t * v2;
}

vec3 operator/(const vec3 &v2, double t) {
    return (1.0 / t) * v2;
}

double dot(const vec3 &v1, const vec3 &v2)  {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

vec3 cross(const vec3 &v1, const vec3 &v2) {
    return vec3(v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0]);  
}
vec3 vec3::unit_vector() const {
    return *this / this->len();
}

vec3 vec3::random() {
    return vec3(randomDouble(), randomDouble(), randomDouble());
}

vec3 vec3::random(const double &min, const double &max) {
    return vec3(randomDouble(min,max), randomDouble(min,max), randomDouble(min,max));
}

vec3 vec3::randomInUnitSphere() {
    while (true) {
        vec3 tmp = vec3::random(-1, 1);
        if (tmp.len_squared() < 1) {
            return tmp;
        }
    }
}

vec3 vec3::randomUnitVectorInSphere() {
    return randomInUnitSphere().unit_vector();
}

vec3 vec3::randomUnitVectorInHemisphere(const vec3 &normal) {
    vec3 vectorInSphere = randomInUnitSphere().unit_vector();
    if (dot(vectorInSphere, normal) > 0.0) {
        return vectorInSphere;
    } else {
        return -vectorInSphere;
    }
}

vec3 vec3::reflect(const vec3 &n) const {
    return *this - 2 * dot(*this, n) * n;
}

vec3 vec3::refract(const vec3 &n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-(*this), n), 1.0);
    vec3 r_out_perp =  etai_over_etat * ((*this) + cos_theta*n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.len_squared())) * n;
    return r_out_perp + r_out_parallel;
}