#include "bvh.hpp"

#include <memory>

#include "aabb.hpp"
#include "hittable_object.hpp"
#include "interval.hpp"

BVHNode::BVHNode(std::vector<std::shared_ptr<HittableObject>>& objects,
                 int start, int end) {
    int axis = rand() % 3;

    auto comparator = (axis == 0)   ? box_x_compare
                      : (axis == 1) ? box_y_compare
                                    : box_z_compare;

    size_t object_span = end - start;

    if (object_span == 1) {
        left = right = objects[start];
    } else if (object_span == 2) {
        if (comparator(objects[start + 1], objects[start]))
            std::swap(objects[start], objects[start + 1]);

        left = objects[start];
        right = objects[start + 1];
    } else {
        std::sort(objects.begin() + start, objects.begin() + end, comparator);

        int mid = start + object_span / 2;

        left = std::make_shared<BVHNode>(objects, start, mid);
        right = std::make_shared<BVHNode>(objects, mid, end);
    }

    bbox = AABB(left->bounding_box(), right->bounding_box());
}

bool BVHNode::hit(const Ray& r, const Interval& ray_t, HitRecord& rec) const {
    if (!bbox.hit(r, ray_t)) {
        return false;
    }

    bool hit_left = left->hit(r, ray_t, rec);
    bool hit_right = right->hit(
        r, Interval(ray_t.min, hit_left ? rec.param : ray_t.max), rec);

    return hit_left || hit_right;
}
