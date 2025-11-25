#pragma once

#include <memory>

#include "aabb.hpp"
#include "hittable_object.hpp"

class Quad : public HittableObject {
protected:
    vec3 _normal;
    std::shared_ptr<Material> _material;
    AABB bbox;
    point3 _base;
    vec3 _w, _u, _v;
    double D;

public:
    Quad(const point3 &base, const vec3 &u, const vec3 &v,
         std::shared_ptr<Material> mat);
    bool hit(const Ray &r, const Interval &ray_inter,
             HitRecord &rec) const override;
    AABB bounding_box() const override;
    virtual bool is_interior(double a, double b, HitRecord &rec) const;
};

class Ellipse : public Quad {
public:
    Ellipse(const point3 &base, const vec3 &u, const vec3 &v,
            std::shared_ptr<Material> mat);
    bool is_interior(double a, double b, HitRecord &rec) const override;
};

class Triangle : public Quad {
public:
    Triangle(const point3 &base, const vec3 &u, const vec3 &v,
             std::shared_ptr<Material> mat);
    bool is_interior(double a, double b, HitRecord &rec) const override;
};