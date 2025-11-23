#pragma once
#include <hittable_object.hpp>
#include <memory>
#include <vector>

#include "aabb.hpp"

class HittableList : public HittableObject {
private:
    std::vector<std::shared_ptr<HittableObject>> _objs;
    AABB bbox;

public:
    HittableList() = default;

    void push_back(std::shared_ptr<HittableObject> obj);
    void clear();

    bool hit(const Ray& r, const Interval& ray_inter,
             HitRecord& rec) const override;

    AABB bounding_box() const override { return bbox; }
    std::vector<std::shared_ptr<HittableObject>> getObjects() const;
    std::vector<std::shared_ptr<HittableObject>>& getObjects();
};