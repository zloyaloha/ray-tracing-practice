#pragma once

#include "aabb.h"
#include "plane.h"
#include "sphere.h"

struct BVHNode {
    AABB box;
    int left;
    int right;
    int type;
};

struct BVHTree {
    BVHNode* nodes;
    int num_nodes;
};

__host__ __device__ inline bool hit_bvh(const Ray& r, Interval ray_t, HitRecord& rec, BVHNode* nodes, int num_nodes,
                                        SphereData* spheres, PlaneData* planes) {
    int stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    bool hit_anything = false;
    float closest = ray_t.max;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];

        if (node_idx >= num_nodes || node_idx < 0) continue;

        BVHNode node = nodes[node_idx];

        if (node.box.hit(r, Interval(ray_t.min, closest))) {
            if (node.left < 0) {
                HitRecord temp_rec;
                bool hit = false;
                if (node.type == 0) {
                    hit = hit_sphere(r, Interval(ray_t.min, closest), temp_rec, spheres[node.right]);
                } else if (node.type == 1) {
                    hit = hit_plane(r, Interval(ray_t.min, closest), temp_rec, planes[node.right]);
                }

                if (hit) {
                    hit_anything = true;
                    closest = temp_rec.t;
                    rec = temp_rec;
                }
            } else {
                if (stack_ptr + 2 <= 32) {
                    int axis = node.type;
                    bool traverse_left_first = r.direction()[axis] >= 0;
                    
                    int first = traverse_left_first ? node.left : node.right;
                    int second = traverse_left_first ? node.right : node.left;

                    stack[stack_ptr++] = second;
                    stack[stack_ptr++] = first;
                }
            }
        }
    }
    return hit_anything;
}
