#include "material.hpp"

class DiffuseLight : public Material {
private:
    color _emit_color;

public:
    DiffuseLight(const color& c) : _emit_color(c) {}
    bool scatter(const Ray& inRay, const HitRecord& rec, color& attenuation,
                 Ray& scattered) const override {
        return false;
    }
    color emitted() const override { return _emit_color; }
};