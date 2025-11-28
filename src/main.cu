#include <cmath>
#include <cstdio>
#include <iostream>

#include "camera.cuh"
#include "materials.cuh"
#include "plane.cuh"
#include "scene.cuh"
#include "sphere.cuh"
#include "bvh_builder.cuh"

#include "scene_params.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Constant memory symbols defined in camera.cu
extern __constant__ CameraData d_cam_data_const;
extern __constant__ SceneData d_scene_data_const;

cudaTextureObject_t load_texture_to_gpu(const char *filename) {
    int width, height, channels;
    float *h_data = stbi_loadf(filename, &width, &height, &channels, 4);
    if (!h_data) {
        std::cerr << "Failed to load texture: " << filename << std::endl;
        return 0;
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, width, height));

    const size_t spitch = width * 4 * sizeof(float);
    checkCudaErrors(
        cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * 4 * sizeof(float), height, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    stbi_image_free(h_data);
    return texObj;
}

CpuTexture* load_texture_cpu(const char *filename) {
    int width, height, channels;
    float *data = stbi_loadf(filename, &width, &height, &channels, 4);
    if (!data) {
        std::cerr << "Failed to load texture: " << filename << std::endl;
        return nullptr;
    }
    return new CpuTexture{data, width, height};
}

void add_cube(std::vector<PlaneData> &planes, const point3 &center, float r, int mat_idx, std::vector<SphereData> &point_lights,
              int light_on_border, int material_idx_border, int edge_light_mat_idx) {
    auto add_tri = [&](const point3 &A, const point3 &B, const point3 &C, const point3 &D) {
        vec3 u = B - A;
        vec3 v = D - A;
        planes.push_back({A, u, v, mat_idx, PlaneType::QUAD});
    };

    const point3 verts_local[8] = {
        point3(-1, -1, -1),  // 0: Левый-Нижний-Задний
        point3(1, -1, -1),   // 1: Правый-Нижний-Задний
        point3(1, 1, -1),    // 2: Правый-Верхний-Задний
        point3(-1, 1, -1),   // 3: Левый-Верхний-Задний
        point3(-1, -1, 1),   // 4: Левый-Нижний-Передний
        point3(1, -1, 1),    // 5: Правый-Нижний-Передний
        point3(1, 1, 1),     // 6: Правый-Верхний-Передний
        point3(-1, 1, 1)     // 7: Левый-Верхний-Передний
    };

    float sphere_radius = r / 100 * 2;

    float dist_to_face = r / std::sqrt(3.0f);

    float light_scale = 1.0f;
    if (dist_to_face > sphere_radius) {
        light_scale = (dist_to_face - sphere_radius) / dist_to_face;
    } else {
        light_scale = 0.0f;
    }

    point3 V[8];
    point3 V_light[8];
    for (int i = 0; i < 8; ++i) {
        vec3 direction = unit_vector(verts_local[i]);

        V[i] = center + direction * r;

        V_light[i] = center + direction * (r * light_scale);
    }

    int edge_pairs[12][2] = {{0, 1}, {1, 5}, {5, 4}, {4, 0}, {3, 2}, {2, 6}, {6, 7}, {7, 3}, {0, 3}, {1, 2}, {5, 6}, {4, 7}};

    for (auto &edge : edge_pairs) {
        point3 start = V_light[edge[0]];
        point3 end = V_light[edge[1]];

        vec3 edge_vec = end - start;
        vec3 mid = (start + end) * 0.5f;
        vec3 radial = unit_vector(mid - center);
        vec3 tangent = unit_vector(cross(edge_vec, radial));
        
        float width = r * 0.05f;
        point3 base = start - tangent * (width * 0.5f);
        vec3 u = edge_vec;
        vec3 v = tangent * width;
        
        planes.push_back({base, u, v, material_idx_border, PlaneType::QUAD});

        for (int i = 0; i < light_on_border; ++i) {
            float t = (i + 0.5f) / light_on_border;
            vec3 pos = (1.0f - t) * start + t * end;
            point_lights.push_back({pos, sphere_radius, edge_light_mat_idx});
        }
    }

    const int faces[6][4] = {
        {4, 5, 6, 7},
        {1, 0, 3, 2},
        {5, 1, 2, 6},
        {4, 7, 3, 0},
        {7, 6, 2, 3},
        {0, 1, 5, 4}
    };

    for (int i = 0; i < 6; ++i) {
        point3 A = V[faces[i][0]];
        point3 B = V[faces[i][1]];
        point3 C = V[faces[i][2]];
        point3 D = V[faces[i][3]];

        add_tri(A, B, C, D);
    }
}

constexpr float phi = 1.61803398875f;
constexpr float inv = 1.0f / phi;

void add_dodecahedron(std::vector<PlaneData> &planes, const point3 &center, float r, int mat_idx,
                      std::vector<SphereData> &point_lights, int light_on_border, int material_idx_border, int edge_light_mat_idx) {
    auto add_tri = [&](const point3 &A, const point3 &B, const point3 &C) {
        vec3 u = B - A;
        vec3 v = C - A;
        planes.push_back({A, u, v, mat_idx, PlaneType::TRIANGLE});
    };

    const point3 verts_local[20] = {
        point3(1, 1, 1),     point3(1, 1, -1),     point3(1, -1, 1),     point3(1, -1, -1),      // 0-3
        point3(-1, 1, 1),    point3(-1, 1, -1),    point3(-1, -1, 1),    point3(-1, -1, -1),     // 4-7
        point3(0, phi, inv), point3(0, phi, -inv), point3(0, -phi, inv), point3(0, -phi, -inv),  // 8-11
        point3(inv, 0, phi), point3(inv, 0, -phi), point3(-inv, 0, phi), point3(-inv, 0, -phi),  // 12-15
        point3(phi, inv, 0), point3(phi, -inv, 0), point3(-phi, inv, 0), point3(-phi, -inv, 0)   // 16-19
    };

    point3 V[20];
    for (int i = 0; i < 20; ++i) {
        V[i] = center + unit_vector(verts_local[i]) * r;
    }

    // Каждая строка — это один пятиугольник.
    const int faces[12][5] = {
        {12, 2, 17, 16, 0},  // Face 1: Правая-Передняя
        {8, 4, 14, 12, 0},   // Face 2: Верхняя-Передняя
        {16, 1, 9, 8, 0},    // Face 3: Верхняя-Правая
        {17, 3, 13, 1, 16},  // Face 4: Правая-Задняя
        {13, 15, 5, 9, 1},   // Face 5: Верхняя-Задняя
        {14, 6, 10, 2, 12},  // Face 6: Нижняя-Передняя
        {10, 11, 3, 17, 2},  // Face 7: Нижняя-Правая
        {3, 11, 7, 15, 13},  // Face 8: Нижняя-Задняя
        {18, 19, 6, 14, 4},  // Face 9: Левая-Передняя
        {9, 5, 18, 4, 8},    // Face 10: Левая-Верхняя
        {7, 11, 10, 6, 19},  // Face 11: Нижняя-Левая
        {5, 15, 7, 19, 18}   // Face 12: Левая-Задняя
    };

    int processed_edges[30][2];
    int edge_count = 0;

    float sphere_radius = r / 100 * 2;
    
    float dist_to_face = r * 0.79465447229f;
    
    float light_scale = 1.0f;
    if (dist_to_face > sphere_radius) {
        light_scale = (dist_to_face - sphere_radius) / dist_to_face;
    }

    point3 V_light[20];
    for (int i = 0; i < 20; ++i) {
        V_light[i] = center + unit_vector(verts_local[i]) * (r * light_scale);
    }

    for (int f = 0; f < 12; ++f) {
        add_tri(V[faces[f][0]], V[faces[f][1]], V[faces[f][2]]);
        add_tri(V[faces[f][0]], V[faces[f][2]], V[faces[f][3]]);
        add_tri(V[faces[f][0]], V[faces[f][3]], V[faces[f][4]]);

        point3 A = V[faces[f][0]];
        point3 B = V[faces[f][1]];
        point3 C = V[faces[f][2]];
        vec3 u = B - A;
        vec3 v = C - A;
        vec3 normal = unit_vector(cross(u, v));
        
        vec3 to_face = unit_vector(A - center);
        float alignment = dot(normal, to_face);
        
        for (int i = 0; i < 5; ++i) {
            int idx1 = faces[f][i];
            int idx2 = faces[f][(i + 1) % 5];
            
            int min_idx = (idx1 < idx2) ? idx1 : idx2;
            int max_idx = (idx1 > idx2) ? idx1 : idx2;

            bool exists = false;
            for(int k=0; k<edge_count; ++k) {
                if (processed_edges[k][0] == min_idx && processed_edges[k][1] == max_idx) {
                    exists = true;
                    break;
                }
            }

            if (!exists) {
                processed_edges[edge_count][0] = min_idx;
                processed_edges[edge_count][1] = max_idx;
                edge_count++;

                point3 start = V_light[min_idx];
                point3 end = V_light[max_idx];

                vec3 edge_vec = end - start;
                vec3 mid = (start + end) * 0.5f;
                vec3 radial = unit_vector(mid - center);
                vec3 tangent = unit_vector(cross(edge_vec, radial));
                
                float width = r * 0.05f;
                point3 base = start - tangent * (width * 0.5f);
                vec3 u = edge_vec;
                vec3 v = tangent * width;
                
                planes.push_back({base, u, v, material_idx_border, PlaneType::QUAD});

                for (int k = 0; k < light_on_border; ++k) {
                    float t = (k + 0.5f) / light_on_border;
                    vec3 pos = (1.0f - t) * start + t * end;
                    point_lights.push_back({pos, sphere_radius, edge_light_mat_idx});
                }
            }
        }
    }
}

void destroy_texture_resources(cudaTextureObject_t texObj) {
    if (texObj == 0) return;

    cudaResourceDesc resDesc;
    checkCudaErrors(cudaGetTextureObjectResourceDesc(&resDesc, texObj));

    checkCudaErrors(cudaDestroyTextureObject(texObj));

    if (resDesc.resType == cudaResourceTypeArray && resDesc.res.array.array != nullptr) {
        checkCudaErrors(cudaFreeArray(resDesc.res.array.array));
    }
}

void add_octahedron(std::vector<PlaneData> &planes, const point3 &center, float r, int mat_idx,
                    std::vector<SphereData> &point_lights, int light_on_border, int material_idx_border, int edge_light_mat_idx) {
    auto add_tri = [&](const point3 &A, const point3 &B, const point3 &C) {
        vec3 u = B - A;
        vec3 v = C - A;
        planes.push_back({A, u, v, mat_idx, PlaneType::TRIANGLE});
    };

    const point3 verts_local[6] = {
        point3(0, 1, 0),   // 0: Top
        point3(0, -1, 0),  // 1: Bot
        point3(0, 0, 1),   // 2: Fwd
        point3(0, 0, -1),  // 3: Bck
        point3(1, 0, 0),   // 4: Rgt
        point3(-1, 0, 0)   // 5: Lft
    };

    point3 V[6];
    point3 V_light[6];
    
    float sphere_radius = r / 100 * 2;
    float dist_to_face = r * 0.57735026919f;
    float light_scale = 1.0f;
    if (dist_to_face > sphere_radius) {
        light_scale = (dist_to_face - sphere_radius) / dist_to_face;
    }

    for (int i = 0; i < 6; ++i) {
        V[i] = center + unit_vector(verts_local[i]) * r;
        V_light[i] = center + unit_vector(verts_local[i]) * (r * light_scale);
    }

    add_tri(V[0], V[2], V[4]);
    add_tri(V[0], V[4], V[3]);
    add_tri(V[0], V[3], V[5]);
    add_tri(V[0], V[5], V[2]);

    add_tri(V[1], V[4], V[2]);
    add_tri(V[1], V[3], V[4]);
    add_tri(V[1], V[5], V[3]);
    add_tri(V[1], V[2], V[5]);

    int edge_pairs[12][2] = {
        {0, 2}, {0, 4}, {0, 3}, {0, 5}, // Top edges
        {1, 2}, {1, 4}, {1, 3}, {1, 5}, // Bot edges
        {2, 4}, {4, 3}, {3, 5}, {5, 2}  // Equator edges
    };

    for (auto &edge : edge_pairs) {
        point3 start = V_light[edge[0]];
        point3 end = V_light[edge[1]];

        vec3 edge_vec = end - start;
        vec3 mid = (start + end) * 0.5f;
        vec3 radial = unit_vector(mid - center);
        vec3 tangent = unit_vector(cross(edge_vec, radial));
        
        float width = r * 0.05f;
        point3 base = start - tangent * (width * 0.5f);
        vec3 u = edge_vec;
        vec3 v = tangent * width;
        
        planes.push_back({base, u, v, material_idx_border, PlaneType::QUAD});

        for (int i = 0; i < light_on_border; ++i) {
            float t = (i + 0.5f) / light_on_border;
            vec3 pos = (1.0f - t) * start + t * end;
            point_lights.push_back({pos, sphere_radius, edge_light_mat_idx});
        }
    }
}

void destroy_all_materials(std::vector<MaterialData> &host_materials) {
    for (MaterialData &mat : host_materials) {
        destroy_texture_resources(mat.tex_obj);
        mat.tex_obj = 0;
        if (mat.cpu_tex) {
            stbi_image_free(mat.cpu_tex->data);
            delete mat.cpu_tex;
            mat.cpu_tex = nullptr;
        }
    }
}

void destroy_scene_arrays(SceneData *host_scene) {
    if (host_scene == nullptr) return;

    if (host_scene->d_spheres != nullptr) {
        cudaFree(host_scene->d_spheres);
        host_scene->d_spheres = nullptr;
    }

    if (host_scene->d_materials != nullptr) {
        cudaFree(host_scene->d_materials);
        host_scene->d_materials = nullptr;
    }

    if (host_scene->d_planes != nullptr) {
        cudaFree(host_scene->d_planes);
        host_scene->d_planes = nullptr;
    }

    if (host_scene->d_bvh_trees != nullptr) {
        cudaFree(host_scene->d_bvh_trees);
        host_scene->d_bvh_trees = nullptr;
    }
}

void create_scene(const SceneParams &scene_params, SceneData &host_scene, std::vector<SphereData> &host_spheres, 
                  std::vector<MaterialData> &host_materials, std::vector<PlaneData> &host_planes, bool use_gpu) {
    MaterialData mat_floor;
    mat_floor.type = METAL;
    mat_floor.albedo = scene_params.floor.tint;
    mat_floor.fuzz = scene_params.floor.reflection_coeff;
    if (!scene_params.floor.texture_path.empty()) {
        if (use_gpu) {
            mat_floor.tex_obj = load_texture_to_gpu(scene_params.floor.texture_path.c_str());
        } else {
            mat_floor.cpu_tex = load_texture_cpu(scene_params.floor.texture_path.c_str());
        }
    }
    host_materials.push_back(mat_floor);
    int floor_mat_idx = host_materials.size() - 1;
    
    MaterialData mat_edge_light;
    mat_edge_light.type = DIFFUSE_LIGHT;
    mat_edge_light.emit = scene_params.lights[0].col * 0.1;
    host_materials.push_back(mat_edge_light);
    int edge_light_mat_idx = host_materials.size() - 1;
    
    for (int i = 0; i < scene_params.bodies.size();++i) {
        BodyParams body = scene_params.bodies[i];
        
        float refl = body.reflection_coeff;
        float trans = body.transparency_coeff;
        
        MaterialData mat_body;
        mat_body.type = DIELECTRIC;
        mat_body.ir = 1.0f + refl;
        float abs_strength = (1.0f - trans) * 0.5f;
        mat_body.absorption = color(
            abs_strength * (1.0f - body.col.x()),
            abs_strength * (1.0f - body.col.y()),
            abs_strength * (1.0f - body.col.z())
        );
        
        host_materials.push_back(mat_body);
        int body_mat_idx = host_materials.size() - 1;
        
        int border_mat_idx = host_materials.size();
        if (i == 0) {
            MaterialData mat_border;
            mat_border.type = METAL;
            mat_border.albedo = color(0.5, 0.5, 0.5);
            mat_border.fuzz = 0.6;
            host_materials.push_back(mat_border);
            add_octahedron(host_planes, body.center, body.radius, body_mat_idx, host_spheres, body.lights_on_edge, border_mat_idx, edge_light_mat_idx);
        } else if (i == 1) {
            MaterialData mat_border;
            mat_border.type = METAL;
            mat_border.albedo = color(0.5, 0.5, 0.5);
            mat_border.fuzz = 0.6;
            host_materials.push_back(mat_border);
            add_cube(host_planes, body.center, body.radius, body_mat_idx, host_spheres, body.lights_on_edge, border_mat_idx, edge_light_mat_idx);
        } else {
            MaterialData mat_border;
            mat_border.type = METAL;
            mat_border.albedo = color(0.5, 0.5, 0.5);
            mat_border.fuzz = 0.6;
            host_materials.push_back(mat_border);
            add_dodecahedron(host_planes, body.center, body.radius, body_mat_idx, host_spheres, body.lights_on_edge, border_mat_idx, edge_light_mat_idx);
        }
    }
    
    vec3 u = scene_params.floor.corners[1] - scene_params.floor.corners[0];
    vec3 v = scene_params.floor.corners[3] - scene_params.floor.corners[0];
    host_planes.push_back({scene_params.floor.corners[0], u, v, floor_mat_idx, PlaneType::QUAD});
    
    for (const auto &light : scene_params.lights) {
        MaterialData mat_point_light;
        mat_point_light.type = DIFFUSE_LIGHT;
        
        mat_point_light.emit = light.col;
        host_materials.push_back(mat_point_light);
        int point_light_mat_idx = host_materials.size() - 1;
        
        host_spheres.push_back({light.position, 1.0f, point_light_mat_idx});
    }
}


SceneParams read_scene_params(std::istream &input) {
    SceneParams params;
    
    input >> params.num_frames;
    input >> params.output_path;
    
    input >> params.width >> params.height >> params.fov_degrees;
    
    input >> params.camera_path.rc0 >> params.camera_path.zc0 >> params.camera_path.phic0;
    input >> params.camera_path.Arc >> params.camera_path.Azc;
    input >> params.camera_path.wrc >> params.camera_path.wzc >> params.camera_path.wc;
    input >> params.camera_path.prc >> params.camera_path.pzc;
    
    input >> params.camera_path.rn0 >> params.camera_path.zn0 >> params.camera_path.phin0;
    input >> params.camera_path.Arn >> params.camera_path.Azn;
    input >> params.camera_path.wrn >> params.camera_path.wzn >> params.camera_path.wn;
    input >> params.camera_path.prn >> params.camera_path.pzn;
    
    int num_bodies = 3;
    params.bodies.resize(num_bodies);
    
    for (int i = 0; i < num_bodies; ++i) {
        BodyParams &body = params.bodies[i];
        input >> body.center.e[0] >> body.center.e[1] >> body.center.e[2];
        input >> body.col.e[0] >> body.col.e[1] >> body.col.e[2];
        input >> body.radius;
        input >> body.reflection_coeff >> body.transparency_coeff;
        input >> body.lights_on_edge;
    }
    
    for (int i = 0; i < 4; ++i) {
        input >> params.floor.corners[i].e[0] 
              >> params.floor.corners[i].e[1] 
              >> params.floor.corners[i].e[2];
    }
    input >> params.floor.texture_path;
    input >> params.floor.tint.e[0] >> params.floor.tint.e[1] >> params.floor.tint.e[2];
    input >> params.floor.reflection_coeff;
    
    int num_lights;
    input >> num_lights;
    if (num_lights > 4) {
        num_lights = 4;
    }
    params.lights.resize(num_lights);
    
    for (int i = 0; i < num_lights; ++i) {
        input >> params.lights[i].position.e[0] 
              >> params.lights[i].position.e[1] 
              >> params.lights[i].position.e[2];
        input >> params.lights[i].col.e[0] 
              >> params.lights[i].col.e[1] 
              >> params.lights[i].col.e[2];
    }
    
    input >> params.render.max_depth >> params.render.sqrt_rays_per_pixel;
    return params;
}

void gpu_render(SceneParams &scene_params) {
    SceneData host_scene;
    std::vector<MaterialData> host_materials;
    std::vector<SphereData> host_spheres;
    std::vector<PlaneData> host_planes;

    create_scene(scene_params, host_scene, host_spheres, host_materials, host_planes, true);

    if (!host_spheres.empty()) {
        cudaMalloc((void **)&host_scene.d_spheres, host_spheres.size() * sizeof(SphereData));
        cudaMemcpy(host_scene.d_spheres, host_spheres.data(), host_spheres.size() * sizeof(SphereData), cudaMemcpyHostToDevice);
    } else {
        host_scene.d_spheres = nullptr;
    }

    if (!host_planes.empty()) {
        cudaMalloc((void **)&host_scene.d_planes, host_planes.size() * sizeof(PlaneData));
        cudaMemcpy(host_scene.d_planes, host_planes.data(), host_planes.size() * sizeof(PlaneData), cudaMemcpyHostToDevice);
    } else {
        host_scene.d_planes = nullptr;
    }

    if (!host_materials.empty()) {
        cudaMalloc((void **)&host_scene.d_materials, host_materials.size() * sizeof(MaterialData));
        cudaMemcpy(host_scene.d_materials, host_materials.data(), host_materials.size() * sizeof(MaterialData),
                   cudaMemcpyHostToDevice);
    } else {
        host_scene.d_materials = nullptr;
    }

    std::vector<BVHNode> host_bvh_nodes = build_bvh(host_spheres, host_planes);
    
    BVHNode* d_bvh_nodes_array = nullptr;
    if (!host_bvh_nodes.empty()) {
        cudaMalloc((void**)&d_bvh_nodes_array, host_bvh_nodes.size() * sizeof(BVHNode));
        cudaMemcpy(d_bvh_nodes_array, host_bvh_nodes.data(), host_bvh_nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
    }
    
    std::vector<BVHTree> host_bvh_trees;
    BVHTree tree;
    tree.nodes = d_bvh_nodes_array; 
    tree.num_nodes = host_bvh_nodes.size();
    host_bvh_trees.push_back(tree);
    host_scene.num_bvh_trees = host_bvh_trees.size();

    if (!host_bvh_trees.empty()) {
        cudaMalloc((void **)&host_scene.d_bvh_trees, host_bvh_trees.size() * sizeof(BVHTree));
        cudaMemcpy(host_scene.d_bvh_trees, host_bvh_trees.data(), host_bvh_trees.size() * sizeof(BVHTree), cudaMemcpyHostToDevice);
    } else {
        host_scene.d_bvh_trees = nullptr;
    }

    host_scene.num_spheres = 0;
    host_scene.num_planes = 0;
    host_scene.num_materials = host_materials.size();

    std::cerr << "Scene created:" << std::endl;
    std::cerr << "  Spheres: " << host_spheres.size() << " (BVH)" << std::endl;
    std::cerr << "  Planes: " << host_planes.size() << " (BVH)" << std::endl;
    std::cerr << "  Materials: " << host_materials.size() << std::endl;
    std::cerr << "  BVH nodes: " << host_bvh_nodes.size() << std::endl;
    
    // Debug: print some light materials
    for (size_t i = 0; i < host_materials.size(); ++i) {
        if (host_materials[i].type == DIFFUSE_LIGHT) {
            std::cerr << "  Light material [" << i << "]: emit=" << host_materials[i].emit.x() << "," << host_materials[i].emit.y() << "," << host_materials[i].emit.z() << std::endl;
        }
    }

    set_scene_data_const(host_scene);

    color *d_fb = nullptr;
    std::size_t num_pixels = static_cast<std::size_t>(scene_params.width) * scene_params.height;
    checkCudaErrors(cudaMalloc(&d_fb, num_pixels * sizeof(color)));

    for (int n = 0; n < scene_params.num_frames; ++n) {
        char filename[256];
        snprintf(filename, sizeof(filename), scene_params.output_path.c_str(), n);
        auto saver = std::make_unique<PNGSaver>(100, filename);
        float t_param = (float(n) / scene_params.num_frames) * 2.0f * M_PI;

        float r_c = scene_params.camera_path.rc0 + scene_params.camera_path.Arc * sin(scene_params.camera_path.wrc * t_param + scene_params.camera_path.prc);
        float z_c = scene_params.camera_path.zc0 + scene_params.camera_path.Azc * sin(scene_params.camera_path.wzc * t_param + scene_params.camera_path.pzc);
        float phi_c = scene_params.camera_path.phic0 + scene_params.camera_path.wc * t_param;
        point3 lookfrom(r_c * cos(phi_c), r_c * sin(phi_c), z_c);

        float r_n = scene_params.camera_path.rn0 + scene_params.camera_path.Arn * sin(scene_params.camera_path.wrn * t_param + scene_params.camera_path.prn);
        float z_n = scene_params.camera_path.zn0 + scene_params.camera_path.Azn * sin(scene_params.camera_path.wzn * t_param + scene_params.camera_path.pzn);
        float phi_n = scene_params.camera_path.phin0 + scene_params.camera_path.wn * t_param;
        point3 lookat(r_n * cos(phi_n), r_n * sin(phi_n), z_n);

        Camera camera(scene_params.height, scene_params.width, std::move(saver), lookfrom, lookat);
        camera.vfov = scene_params.fov_degrees;
        int sqrt_spp = scene_params.render.sqrt_rays_per_pixel;
        camera.samplesPerPixel = sqrt_spp * sqrt_spp;
        camera.maxDepth = scene_params.render.max_depth;
        camera.background_color = color(0, 0, 0);

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        checkCudaErrors(cudaEventRecord(start));
        camera.render(&host_scene, d_fb);
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float time;
        checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
        checkCudaErrors(cudaEventDestroy(start));
        checkCudaErrors(cudaEventDestroy(stop));
        long long total_rays = (long long)scene_params.width * scene_params.height * scene_params.render.sqrt_rays_per_pixel * scene_params.render.sqrt_rays_per_pixel;
        std::cout << n << "\t" << time << "\t" << total_rays << "\n";
    }
    destroy_scene_arrays(&host_scene);
    destroy_all_materials(host_materials);
    cudaFree(d_fb);
}

void print_default_config() {
    std::cout << 100 << "\n";
    std::cout << "/home/checker/pgp/ivanov/kp/out/img_%d.data\n";
    std::cout << 1080 << " " << 720 << " " << 50 << "\n";
    std::cout << 15.0 << " " << 4.5 << " " << 3.14159 << "\t";
    std::cout << 0.0 << " " << 4.5 << "\t";
    std::cout << 0.0 << " " << 1.0 << " " << 1.0 << "\t";
    std::cout << 0.0 << " " << -1.57 << "\n";
    std::cout << 0.0 << " " << 4.5 << "\t";
    std::cout << 0.0 << " " << 1.0 << " " << 0.0 << "\t";
    std::cout << 0.0 << " " << -1.57 << "\n";
    std::cout << 0.0 << " " << 0.0 << " " << 3.0 << "\t";
    std::cout << 0.3 << " " << 0.0 << " " << 0.0 << "\t";
    std::cout << 3.0 << " " << 1.5 << " " << 0.1 << "\t";
    std::cout << 3 << "\n";
    std::cout << 4 << " " << 0.0 << " " << 6.0 << "\t";
    std::cout << 0.0 << " " << 0.3 << " " << 0.0 << "\t";
    std::cout << 3.0 << " " << 1.2 << " " << 0.1 << "\t";
    std::cout << 2 << "\n";
    std::cout << 8 << " " << 0.0 << " " << 9.0 << "\t";
    std::cout << 0.0 << " " << 0.0 << " " << 0.3 << "\t";
    std::cout << 3.0 << " " << 1 << " " << 0.1 << "\t";
    std::cout << 1 << "\n";
    std::cout << -15.0 << " " << -15.0 << " " << -1.0 << "\t";
    std::cout << -15.0 << " " << 15.0 << " " << -1.0 << "\t";
    std::cout << 15.0 << " " << 15.0 << " " << -1.0 << "\t";
    std::cout << 15.0 << " " << -15.0 << " " << -1.0 << "\t";
    std::cout << "../floor2.jpg\n";
    std::cout << 1.0 << " " << 1.0 << " " << 1.0 << "\n";
    std::cout << 0.3 << "\n";
    std::cout << 4 << "\n";
    std::cout << -15.0 << " " << -15.0 << " " << 1 << "\t";
    std::cout << 10.0 << " " << 10.0 << " " << 10.0 << "\n";
    std::cout << -15.0 << " " << 15.0 << " " << 1 << "\t";
    std::cout << 10.0 << " " << 10.0 << " " << 10.0 << "\n";
    std::cout << 15.0 << " " << 15.0 << " " << 1 << "\t";
    std::cout << 10.0 << " " << 10.0 << " " << 10.0 << "\n";
    std::cout << 15.0 << " " << -15.0 << " " << 1 << "\t";
    std::cout << 10.0 << " " << 10.0 << " " << 10.0 << "\n";
    std::cout << 50 << " " << 50 << "\n";
}

void cpu_render(SceneParams scene_params) {
    SceneData host_scene;
    std::vector<MaterialData> host_materials;
    std::vector<SphereData> host_spheres;
    std::vector<PlaneData> host_planes;

    create_scene(scene_params, host_scene, host_spheres, host_materials, host_planes, false);

    // No device allocation needed for CPU render, but we need pointers in host_scene
    // However, SceneData stores pointers. For CPU render, we can just point to the vectors' data?
    // Wait, SceneData has d_spheres, d_materials etc. which are pointers.
    // The render_kernel (host version) takes SceneData.
    // We can just cast host pointers to the types expected by SceneData (which are just pointers).
    // Actually, SceneData members are defined as `SphereData* d_spheres`.
    // So we can assign host_spheres.data() to d_spheres.

    host_scene.d_spheres = host_spheres.data();
    host_scene.d_planes = host_planes.data();
    host_scene.d_materials = host_materials.data();

    std::vector<BVHNode> host_bvh_nodes = build_bvh(host_spheres, host_planes);
    
    BVHTree tree;
    tree.nodes = host_bvh_nodes.data(); 
    tree.num_nodes = host_bvh_nodes.size();
    
    // We need a list of trees, even if just one
    std::vector<BVHTree> host_bvh_trees;
    host_bvh_trees.push_back(tree);
    
    host_scene.d_bvh_trees = host_bvh_trees.data();
    host_scene.num_bvh_trees = host_bvh_trees.size();

    host_scene.num_spheres = 0; // BVH handles everything
    host_scene.num_planes = 0;
    host_scene.num_materials = host_materials.size();

    // Framebuffer
    std::vector<color> framebuffer(scene_params.width * scene_params.height);

    for (int n = 0; n < scene_params.num_frames; ++n) {
        char filename[256];
        snprintf(filename, sizeof(filename), scene_params.output_path.c_str(), n);
        auto saver = std::make_unique<PNGSaver>(100, filename);
        float t_param = (float(n) / scene_params.num_frames) * 2.0f * M_PI;

        float r_c = scene_params.camera_path.rc0 + scene_params.camera_path.Arc * sin(scene_params.camera_path.wrc * t_param + scene_params.camera_path.prc);
        float z_c = scene_params.camera_path.zc0 + scene_params.camera_path.Azc * sin(scene_params.camera_path.wzc * t_param + scene_params.camera_path.pzc);
        float phi_c = scene_params.camera_path.phic0 + scene_params.camera_path.wc * t_param;
        point3 lookfrom(r_c * cos(phi_c), r_c * sin(phi_c), z_c);

        float r_n = scene_params.camera_path.rn0 + scene_params.camera_path.Arn * sin(scene_params.camera_path.wrn * t_param + scene_params.camera_path.prn);
        float z_n = scene_params.camera_path.zn0 + scene_params.camera_path.Azn * sin(scene_params.camera_path.wzn * t_param + scene_params.camera_path.pzn);
        float phi_n = scene_params.camera_path.phin0 + scene_params.camera_path.wn * t_param;
        point3 lookat(r_n * cos(phi_n), r_n * sin(phi_n), z_n);

        Camera camera(scene_params.height, scene_params.width, std::move(saver), lookfrom, lookat);
        camera.vfov = scene_params.fov_degrees;
        int sqrt_spp = scene_params.render.sqrt_rays_per_pixel;
        camera.samplesPerPixel = sqrt_spp * sqrt_spp;
        camera.maxDepth = scene_params.render.max_depth;
        camera.background_color = color(0, 0, 0);

        CameraData cam_data;
        
        float theta = camera.vfov * M_PI / 180.0;
        float h = tan(theta / 2);
        float viewport_height = 2.0 * h;
        float viewport_width = viewport_height * (static_cast<float>(camera.imageWidth) / camera.imageHeight);

        vec3 w = unit_vector(camera.origin - camera.look_at);
        vec3 u = unit_vector(cross(vec3(0,0,1), w)); // vup is hardcoded (0,0,1) in constructor
        vec3 v = cross(w, u);

        vec3 horizontal = viewport_width * u;
        vec3 vertical = viewport_height * v;

        cam_data.origin = camera.origin;
        cam_data.pixel_delta_u = horizontal / camera.imageWidth;
        cam_data.pixel_delta_v = -vertical / camera.imageHeight;
        point3 upper_left = camera.origin - w - horizontal / 2 + vertical / 2;
        cam_data.pixel00_loc = upper_left + 0.5 * (cam_data.pixel_delta_u + cam_data.pixel_delta_v);
        cam_data.background = camera.background_color;
        cam_data.image_width = camera.imageWidth;
        cam_data.image_height = camera.imageHeight;
        cam_data.samples_per_pixel = camera.samplesPerPixel;
        cam_data.max_depth = camera.maxDepth;

        render_kernel_cpu(framebuffer.data(), cam_data, host_scene);
        
        for (int j = 0; j < camera.imageHeight; ++j) {
            for (int i = 0; i < camera.imageWidth; ++i) {
                camera.saver->writeColor(framebuffer[static_cast<std::size_t>(j) * camera.imageWidth + i]);
            }
        }
        
        std::cout << "Frame " << n << " rendered." << std::endl;
    }
    
    destroy_all_materials(host_materials);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        SceneParams scene_params = read_scene_params(std::cin);
        gpu_render(scene_params);
        return 0;
    }

    std::string arg1 = argv[1];
    if (arg1 == "--gpu") {
        SceneParams scene_params = read_scene_params(std::cin);
        gpu_render(scene_params);
    } else if (arg1 == "--cpu") {
        std::cerr << "Running CPU render..." << std::endl;
        SceneParams scene_params = read_scene_params(std::cin);
        cpu_render(scene_params);
    } else if (arg1 == "--default") {
        print_default_config();
    }
    return 0;
}