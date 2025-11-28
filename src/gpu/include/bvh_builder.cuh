#pragma once

#include <vector>
#include <algorithm>
#include "bvh.cuh"
#include "sphere.cuh"
#include "plane.cuh"

struct BVHPrimitive {
    AABB box;
    int type;
    int index;
    point3 centroid;
};

__host__ inline AABB get_sphere_aabb(const SphereData& s) {
    vec3 rvec(s.radius, s.radius, s.radius);
    return AABB(s.center - rvec, s.center + rvec);
}

__host__ inline AABB get_plane_aabb(const PlaneData& p) {
    point3 p0 = p.base;
    point3 p1 = p.base + p.u;
    point3 p2 = p.base + p.v;
    point3 p3 = p.base + p.u + p.v;
    
    AABB box(p0, p1);
    box = surround(box, AABB(p2, p3));
    
    float min_x = fminf(p0.x(), fminf(p1.x(), p2.x()));
    float max_x = fmaxf(p0.x(), fmaxf(p1.x(), p2.x()));
    float min_y = fminf(p0.y(), fminf(p1.y(), p2.y()));
    float max_y = fmaxf(p0.y(), fmaxf(p1.y(), p2.y()));
    float min_z = fminf(p0.z(), fminf(p1.z(), p2.z()));
    float max_z = fmaxf(p0.z(), fmaxf(p1.z(), p2.z()));

    if (p.type == QUAD || p.type == ELLIPSE) {
        min_x = fminf(min_x, p3.x()); max_x = fmaxf(max_x, p3.x());
        min_y = fminf(min_y, p3.y()); max_y = fmaxf(max_y, p3.y());
        min_z = fminf(min_z, p3.z()); max_z = fmaxf(max_z, p3.z());
    }
    
    AABB res(point3(min_x, min_y, min_z), point3(max_x, max_y, max_z));
    res.pad();
    return res;
}

__host__ int build_bvh_recursive(std::vector<BVHPrimitive>& primitives, int start, int end, std::vector<BVHNode>& nodes) {
    int node_idx = nodes.size();
    nodes.push_back({});

    AABB box;
    if (start < end) box = primitives[start].box;
    for (int i = start + 1; i < end; ++i) {
        box = surround(box, primitives[i].box);
    }
    nodes[node_idx].box = box;

    int span = end - start;
    if (span == 1) {
        nodes[node_idx].left = -1;
        nodes[node_idx].right = primitives[start].index;
        nodes[node_idx].type = primitives[start].type;
    } else {
        AABB centroid_box;
        if (start < end) centroid_box = AABB(primitives[start].centroid, primitives[start].centroid);
        for (int i = start + 1; i < end; ++i) {
            centroid_box = surround(centroid_box, AABB(primitives[i].centroid, primitives[i].centroid));
        }
        
        int axis = 0;
        float max_extent = centroid_box.x_interval.size();
        if (centroid_box.y_interval.size() > max_extent) { axis = 1; max_extent = centroid_box.y_interval.size(); }
        if (centroid_box.z_interval.size() > max_extent) { axis = 2; }

        int mid = (start + end) / 2;
        std::nth_element(primitives.begin() + start, primitives.begin() + mid, primitives.begin() + end,
            [axis](const BVHPrimitive& a, const BVHPrimitive& b) {
                return a.centroid[axis] < b.centroid[axis];
            });
        
        int left_child = build_bvh_recursive(primitives, start, mid, nodes);
        int right_child = build_bvh_recursive(primitives, mid, end, nodes);
        
        nodes[node_idx].left = left_child;
        nodes[node_idx].right = right_child;
        nodes[node_idx].type = -1; 
    }
    return node_idx;
}

__host__ std::vector<BVHNode> build_bvh(const std::vector<SphereData>& spheres, const std::vector<PlaneData>& planes) {
    std::vector<BVHPrimitive> primitives;
    primitives.reserve(spheres.size() + planes.size());

    for (size_t i = 0; i < spheres.size(); ++i) {
        AABB box = get_sphere_aabb(spheres[i]);
        point3 centroid = spheres[i].center;
        primitives.push_back({box, 0, (int)i, centroid});
    }

    for (size_t i = 0; i < planes.size(); ++i) {
        AABB box = get_plane_aabb(planes[i]);
        point3 centroid = planes[i].base + (planes[i].u + planes[i].v) * 0.5f; // Approx centroid
        primitives.push_back({box, 1, (int)i, centroid});
    }

    std::vector<BVHNode> nodes;
    if (primitives.empty()) return nodes;

    build_bvh_recursive(primitives, 0, primitives.size(), nodes);
    return nodes;
}
