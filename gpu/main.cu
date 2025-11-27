#include <cmath>
#include <iostream>

#include "camera.cuh"
#include "materials.cuh"
#include "plane.cuh"
#include "scene.cuh"
#include "sphere.cuh"

struct CameraPathParams {
    float rc0, zc0, phic0;
    float Arc, Azc;
    float wrc, wzc, wc;
    float prc, pzc;
    
    float rn0, zn0, phin0;
    float Arn, Azn;
    float wrn, wzn, wn;
    float prn, pzn;
};

struct BodyParams {
    point3 center;
    color col;
    float radius;
    float reflection_coeff;
    float transparency_coeff;
    int lights_on_edge;
};

struct FloorParams {
    point3 corners[4];
    std::string texture_path;
    color tint;
    float reflection_coeff;
};

struct LightSourceParams {
    point3 position;
    color col;
};

struct RenderParams {
    int max_depth;
    int sqrt_rays_per_pixel;
};

struct SceneParams {
    int num_frames;
    std::string output_path;
    int width;
    int height;
    float fov_degrees;
    
    CameraPathParams camera_path;
    std::vector<BodyParams> bodies;
    FloorParams floor;
    std::vector<LightSourceParams> lights;
    RenderParams render;
};

__constant__ SceneData d_scene_data_const;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // Скачайте этот хедер

// Функция возвращает созданный объект текстуры
cudaTextureObject_t load_texture_to_gpu(const char *filename) {
    int width, height, channels;
    // Загружаем как float для точности
    float *h_data = stbi_loadf(filename, &width, &height, &channels, 4);
    if (!h_data) {
        std::cerr << "Failed to load texture: " << filename << std::endl;
        return 0;
    }

    // 1. Выделяем CUDA Array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, width, height));

    // 2. Копируем данные
    const size_t spitch = width * 4 * sizeof(float);
    checkCudaErrors(
        cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * 4 * sizeof(float), height, cudaMemcpyHostToDevice));

    // 3. Настраиваем параметры ресурса
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // 4. Настраиваем параметры текстуры (фильтрация, повторение)
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;  // Повторять текстуру
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;   // Линейная интерполяция!
    texDesc.readMode = cudaReadModeElementType;  // Читаем как float
    texDesc.normalizedCoords = 1;                // Используем координаты [0, 1]

    // 5. Создаем объект
    cudaTextureObject_t texObj = 0;
    checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    stbi_image_free(h_data);
    return texObj;
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

    point3 V[8];        // Геометрия
    point3 V_light[8];  // Опорные точки для света

    float inset = sphere_radius;
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
        
        planes.push_back({base, u, v, material_idx_border, PlaneType::QUAD}); // Material 1 (Metal)

        for (int i = 0; i < light_on_border; ++i) {
            float t = (i + 0.5f) / light_on_border;
            vec3 pos = (1.0f - t) * start + t * end;
            point_lights.push_back({pos, sphere_radius, edge_light_mat_idx});
        }
    }

    const int faces[6][4] = {
        {4, 5, 6, 7},  // Z+: (0, 0, 1)
        {1, 0, 3, 2},  // Z-: (0, 0, -1)
        {5, 1, 2, 6},  // X+: (1, 0, 0)
        {4, 7, 3, 0},  // X-: (-1, 0, 0)
        {7, 6, 2, 3},  // Y+: (0, 1, 0)
        {0, 1, 5, 4}   // Y- (нижняя)
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
}

void create_scene(const SceneParams &scene_params, SceneData &host_scene, std::vector<SphereData> &host_spheres, 
                  std::vector<MaterialData> &host_materials, std::vector<PlaneData> &host_planes) {
    MaterialData mat_floor;
    mat_floor.type = METAL;
    mat_floor.albedo = scene_params.floor.tint;
    mat_floor.fuzz = scene_params.floor.reflection_coeff;
    if (!scene_params.floor.texture_path.empty()) {
        mat_floor.tex_obj = load_texture_to_gpu(scene_params.floor.texture_path.c_str());
    }
    host_materials.push_back(mat_floor);
    int floor_mat_idx = host_materials.size() - 1;
    
    // Material for edge light spheres
    MaterialData mat_edge_light;
    mat_edge_light.type = DIFFUSE_LIGHT;
    mat_edge_light.emit = color(0.5f, 0.5f, 0.5f);
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
        std::cout << "Light source emission: " << light.col << std::endl;
        host_materials.push_back(mat_point_light);
        int point_light_mat_idx = host_materials.size() - 1;
        
        host_spheres.push_back({light.position, 1.0f, point_light_mat_idx});
    }
}


SceneParams read_scene_params(std::istream &input) {
    SceneParams params;
    
    input >> params.num_frames;
    input >> params.output_path;
    std::cout << "Number of frames: " << params.num_frames << std::endl;  
    std::cout << "Save path: " << params.output_path << std::endl;  
    
    input >> params.width >> params.height >> params.fov_degrees;
    std::cout << "Resolution: " << params.width << "x" << params.height << std::endl;
    
    input >> params.camera_path.rc0 >> params.camera_path.zc0 >> params.camera_path.phic0;
    std::cout << "Camera path: " << params.camera_path.rc0 << " " << params.camera_path.zc0 << " " << params.camera_path.phic0 << std::endl;
    input >> params.camera_path.Arc >> params.camera_path.Azc;
    std::cout << "Camera path: " << params.camera_path.Arc << " " << params.camera_path.Azc << std::endl;
    input >> params.camera_path.wrc >> params.camera_path.wzc >> params.camera_path.wc;
    std::cout << "Camera path: " << params.camera_path.wrc << " " << params.camera_path.wzc << " " << params.camera_path.wc << std::endl;
    input >> params.camera_path.prc >> params.camera_path.pzc;
    std::cout << "Camera path: " << params.camera_path.prc << " " << params.camera_path.pzc << std::endl;
    
    input >> params.camera_path.rn0 >> params.camera_path.zn0 >> params.camera_path.phin0;
    std::cout << "Camera path: " << params.camera_path.rn0 << " " << params.camera_path.zn0 << " " << params.camera_path.phin0 << std::endl;
    input >> params.camera_path.Arn >> params.camera_path.Azn;
    std::cout << "Camera path: " << params.camera_path.Arn << " " << params.camera_path.Azn << std::endl;
    input >> params.camera_path.wrn >> params.camera_path.wzn >> params.camera_path.wn;
    std::cout << "Camera path: " << params.camera_path.wrn << " " << params.camera_path.wzn << " " << params.camera_path.wn << std::endl;
    input >> params.camera_path.prn >> params.camera_path.pzn;
    std::cout << "Camera path: " << params.camera_path.prn << " " << params.camera_path.pzn << std::endl;
    
    int num_bodies = 3;
    params.bodies.resize(num_bodies);
    std::cout << "Number of bodies: " << num_bodies << std::endl;
    
    for (int i = 0; i < num_bodies; ++i) {
        BodyParams &body = params.bodies[i];
        input >> body.center.e[0] >> body.center.e[1] >> body.center.e[2];
        std::cout << "Center " << i << ": " << body.center << std::endl;
        input >> body.col.e[0] >> body.col.e[1] >> body.col.e[2];
        std::cout << "Color " << i << ": " << body.col << std::endl;
        input >> body.radius;
        std::cout << "Radius " << i << ": " << body.radius << std::endl;
        input >> body.reflection_coeff >> body.transparency_coeff;
        std::cout << "Reflection " << i << ": " << body.reflection_coeff << " Transparency " << body.transparency_coeff << std::endl;
        input >> body.lights_on_edge;
        std::cout << "Lights on edge " << i << ": " << body.lights_on_edge << std::endl;
    }
    
    for (int i = 0; i < 4; ++i) {
        input >> params.floor.corners[i].e[0] 
              >> params.floor.corners[i].e[1] 
              >> params.floor.corners[i].e[2];
        std::cout << "Corner " << i << ": " << params.floor.corners[i] << std::endl;
    }
    input >> params.floor.texture_path;
    std::cout << "Texture path: " << params.floor.texture_path << std::endl;
    input >> params.floor.tint.e[0] >> params.floor.tint.e[1] >> params.floor.tint.e[2];
    std::cout << "Tint: " << params.floor.tint << std::endl;
    input >> params.floor.reflection_coeff;
    std::cout << "Reflection: " << params.floor.reflection_coeff << std::endl;
    
    int num_lights;
    input >> num_lights;
    std::cout << "Number of lights: " << num_lights << std::endl;
    if (num_lights > 4) {
        std::cerr << "Warning: More than 4 lights specified (" << num_lights << "). Using only first 4.\n";
        num_lights = 4;
    }
    params.lights.resize(num_lights);
    
    for (int i = 0; i < num_lights; ++i) {
        input >> params.lights[i].position.e[0] 
              >> params.lights[i].position.e[1] 
              >> params.lights[i].position.e[2];
        std::cout << "Position " << i << ": " << params.lights[i].position << std::endl;
        input >> params.lights[i].col.e[0] 
              >> params.lights[i].col.e[1] 
              >> params.lights[i].col.e[2];
        std::cout << "Color " << i << ": " << params.lights[i].col << std::endl;
    }
    
    input >> params.render.max_depth >> params.render.sqrt_rays_per_pixel;
    std::cout << "Max depth: " << params.render.max_depth << std::endl;
    std::cout << "Square root of rays per pixel: " << params.render.sqrt_rays_per_pixel << std::endl;
    
    return params;
}

int main() {
    SceneParams scene_params = read_scene_params(std::cin);

    SceneData host_scene;
    std::vector<MaterialData> host_materials;
    std::vector<SphereData> host_spheres;
    std::vector<PlaneData> host_planes;

    create_scene(scene_params, host_scene, host_spheres, host_materials, host_planes);  // Заполняем CPU векторы

    host_scene.num_spheres = host_spheres.size();
    host_scene.num_planes = host_planes.size();
    host_scene.num_materials = host_materials.size();

    cudaMalloc((void **)&host_scene.d_materials, host_materials.size() * sizeof(MaterialData));
    cudaMemcpy(host_scene.d_materials, host_materials.data(), host_materials.size() * sizeof(MaterialData),
               cudaMemcpyHostToDevice);

    cudaMalloc((void **)&host_scene.d_spheres, host_spheres.size() * sizeof(SphereData));
    cudaMemcpy(host_scene.d_spheres, host_spheres.data(), host_spheres.size() * sizeof(SphereData), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&host_scene.d_planes, host_planes.size() * sizeof(PlaneData));
    cudaMemcpy(host_scene.d_planes, host_planes.data(), host_planes.size() * sizeof(PlaneData), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_scene_data_const, &host_scene, sizeof(SceneData), 0, cudaMemcpyHostToDevice);

    color *d_fb = nullptr;
    std::size_t num_pixels = static_cast<std::size_t>(scene_params.width) * scene_params.height;
    checkCudaErrors(cudaMalloc(&d_fb, num_pixels * sizeof(color)));

    for (int n = 0; n < scene_params.num_frames; ++n) {
        auto saver = std::make_unique<PNGSaver>(100, "../images/render" + std::to_string(n) + ".png");
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
        if (camera.maxDepth <= 0 || camera.maxDepth > 100) camera.maxDepth = 50;
        camera.background_color = color(0, 0, 0);
        
        if (n == 0) {
            std::cout << "Camera from: " << lookfrom << " looking at: " << lookat << std::endl;
            std::cout << "Image width: " << camera.imageWidth << ", Aspect ratio: " << camera.aspectRatio << std::endl;
            std::cout << "FOV: " << camera.vfov << " degrees" << std::endl;
            std::cout << "Samples per pixel: " << camera.samplesPerPixel << ", Max depth: " << camera.maxDepth << std::endl;
            std::cout << "Num spheres: " << host_spheres.size() << ", Num planes: " << host_planes.size() << std::endl;
        }

        cudaMemcpy(host_scene.d_planes, host_planes.data(), host_planes.size() * sizeof(PlaneData), cudaMemcpyHostToDevice);
        cudaMemcpy(host_scene.d_spheres, host_spheres.data(), host_spheres.size() * sizeof(SphereData), cudaMemcpyHostToDevice);

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
        std::cout << "Frame generated in " << time << " ms" << std::endl;
    }
    destroy_scene_arrays(&host_scene);
    destroy_all_materials(host_materials);
    cudaFree(d_fb);

    std::cout << "Rendering completed. Output saved to render_gpu.png\n";
    return 0;
}