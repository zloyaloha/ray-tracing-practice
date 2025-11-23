#pragma once

#include <future>

#include "aabb.hpp"
#include "hittable_list.hpp"
#include "hittable_object.hpp"
#include "interval.hpp"

class BVHNode : public HittableObject {
public:
    BVHNode(HittableList list)
        : BVHNode(list.getObjects(), 0, list.getObjects().size()) {}

    BVHNode(std::vector<std::shared_ptr<HittableObject>>& objects, int start,
            int end);

    bool hit(const Ray& r, const Interval& ray_t,
             HitRecord& rec) const override;
    AABB bounding_box() const override { return bbox; }

    static bool box_x_compare(const std::shared_ptr<HittableObject>& a,
                              const std::shared_ptr<HittableObject>& b) {
        return a->bounding_box().getX().min < b->bounding_box().getX().min;
    }

    static bool box_y_compare(const std::shared_ptr<HittableObject>& a,
                              const std::shared_ptr<HittableObject>& b) {
        return a->bounding_box().getY().min < b->bounding_box().getY().min;
    }

    static bool box_z_compare(const std::shared_ptr<HittableObject>& a,
                              const std::shared_ptr<HittableObject>& b) {
        return a->bounding_box().getZ().min < b->bounding_box().getZ().min;
    }

private:
    std::shared_ptr<HittableObject> left;
    std::shared_ptr<HittableObject> right;
    AABB bbox;
};