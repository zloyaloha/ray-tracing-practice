#include <cmath>
#include <iostream>
#include <memory>

#include "camera.cuh"
#include "materials.cuh"
#include "plane.cuh"
#include "scene.cuh"
#include "sphere.cuh"

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
              int light_on_border) {
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

    float sphere_radius = 0.01f;

    float dist_to_face = r / std::sqrt(3.0f);

    float light_scale = 1.0f;
    if (dist_to_face > sphere_radius) {
        light_scale = (dist_to_face - sphere_radius) / dist_to_face;
    } else {
        light_scale = 0.0f;
    }

    point3 V[8];        // Геометрия
    point3 V_light[8];  // Опорные точки для света

    for (int i = 0; i < 8; ++i) {
        vec3 direction = unit_vector(verts_local[i]);

        V[i] = center + direction * r;

        V_light[i] = center + direction * (r * light_scale);

        point_lights.push_back({V_light[i], sphere_radius, 3});
    }

    // Пары индексов для ребер
    int edge_pairs[12][2] = {{0, 1}, {1, 5}, {5, 4}, {4, 0}, {3, 2}, {2, 6}, {6, 7}, {7, 3}, {0, 3}, {1, 2}, {5, 6}, {4, 7}};

    int n = 5;  // Сфер на ребре

    for (auto &edge : edge_pairs) {
        point3 start = V_light[edge[0]];
        point3 end = V_light[edge[1]];

        for (int i = 0; i < n; ++i) {
            float t = (i + 0.5f) / n;
            vec3 pos = (1.0f - t) * start + t * end;
            point_lights.push_back({pos, sphere_radius, 6});
        }
    }

    const int faces[6][4] = {
        {7, 6, 5, 4},  // Передняя (Z+)
        {6, 2, 1, 5},  // Правая   (X+)
        {2, 3, 0, 1},  // Задняя   (Z-)
        {3, 7, 4, 0},  // Левая    (X-)
        {3, 2, 6, 7},  // Верхняя  (Y+)
        {4, 5, 1, 0}   // Нижняя   (Y-)
    };

    for (int i = 0; i < 6; ++i) {
        add_tri(V[faces[i][0]], V[faces[i][1]], V[faces[i][2]], V[faces[i][3]]);
    }
}

constexpr float phi = 1.61803398875f;
constexpr float inv = 1.0f / phi;

void add_dodecahedron(std::vector<PlaneData> &planes, const point3 &center, float r, int mat_idx) {
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
        {0, 16, 17, 2, 12},  // Face 1: Правая-Передняя
        {0, 12, 14, 4, 8},   // Face 2: Верхняя-Передняя
        {0, 8, 9, 1, 16},    // Face 3: Верхняя-Правая
        {16, 1, 13, 3, 17},  // Face 4: Правая-Задняя
        {1, 9, 5, 15, 13},   // Face 5: Верхняя-Задняя
        {12, 2, 10, 6, 14},  // Face 6: Нижняя-Передняя
        {2, 17, 3, 11, 10},  // Face 7: Нижняя-Правая
        {13, 15, 7, 11, 3},  // Face 8: Нижняя-Задняя
        {4, 14, 6, 19, 18},  // Face 9: Левая-Передняя
        {8, 4, 18, 5, 9},    // Face 10: Левая-Верхняя
        {19, 6, 10, 11, 7},  // Face 11: Нижняя-Левая
        {18, 19, 7, 15, 5}   // Face 12: Левая-Задняя
    };

    for (int f = 0; f < 12; ++f) {
        add_tri(V[faces[f][0]], V[faces[f][1]], V[faces[f][2]]);
        add_tri(V[faces[f][0]], V[faces[f][2]], V[faces[f][3]]);
        add_tri(V[faces[f][0]], V[faces[f][3]], V[faces[f][4]]);
    }
}

void add_octahedron(std::vector<PlaneData> &planes, const point3 &center, float r, int mat_idx) {
    auto top = center + vec3(0, r, 0);
    auto bot = center + vec3(0, -r, 0);

    auto fwd = center + vec3(0, 0, r);
    auto bck = center + vec3(0, 0, -r);

    auto rgt = center + vec3(r, 0, 0);
    auto lft = center + vec3(-r, 0, 0);

    auto add_tri = [&](const point3 &A, const point3 &B, const point3 &C) {
        std::cout << std::endl;
        vec3 u = B - A;
        vec3 v = C - A;
        planes.push_back({A, u, v, mat_idx, PlaneType::TRIANGLE});
    };

    add_tri(top, rgt, fwd);  // Передняя правая (+x, +z)
    add_tri(top, bck, rgt);  // Задняя правая (-z, +x)
    add_tri(top, lft, bck);  // Задняя левая (-x, -z)
    add_tri(top, fwd, lft);  // Передняя левая (+z, -x)

    add_tri(bot, fwd, rgt);  // Передняя правая
    add_tri(bot, rgt, bck);  // Задняя правая
    add_tri(bot, bck, lft);  // Задняя левая
    add_tri(bot, lft, fwd);  // Передняя левая
}

void create_scene(std::vector<MaterialData> &host_materials, std::vector<SphereData> &host_spheres,
                  std::vector<PlaneData> &host_planes, std::vector<PointLightData> &host_point_lights) {
    // 0: Материал для земли
    MaterialData mat_ground;
    mat_ground.type = LAMBERTIAN;
    mat_ground.albedo = color(0.8f, 0.8f, 0.0f);
    mat_ground.tex_obj = load_texture_to_gpu("../floor.jpeg");

    // 1: Металл
    MaterialData mat_metal;
    mat_metal.type = METAL;
    mat_metal.albedo = color(1, 0, 0.8f);
    mat_metal.fuzz = 0.2f;

    // 2: Диэлектрик (стекло)
    MaterialData mat_glass;
    mat_glass.type = DIELECTRIC;
    mat_glass.ir = 2.0f;
    mat_glass.absorption = vec3(0.0, 0.0, 0.5);

    // 3: Диффузный свет
    MaterialData mat_light;
    mat_light.type = DIFFUSE_LIGHT;
    mat_light.emit = color(0.2f, 0.2f, 0.2f);

    MaterialData mat_light_bright;
    mat_light_bright.type = DIFFUSE_LIGHT;
    mat_light_bright.emit = color(1.f, 1.f, 1.f);

    MaterialData mat_sphere1;
    mat_sphere1.type = LAMBERTIAN;
    mat_sphere1.albedo = color(1.f, 0.1f, 1.f);
    mat_sphere1.tex_obj = load_texture_to_gpu("../images.png");

    MaterialData mat_metal_ground;
    mat_metal_ground.type = METAL;
    mat_metal_ground.albedo = color(0.1f, 0.1f, 0.1f);
    mat_metal_ground.fuzz = 0.01f;
    mat_metal_ground.tex_obj = load_texture_to_gpu("../floor2.jpg");

    host_materials.push_back(mat_ground);
    host_materials.push_back(mat_metal);
    host_materials.push_back(mat_glass);
    host_materials.push_back(mat_light);
    host_materials.push_back(mat_sphere1);
    host_materials.push_back(mat_metal_ground);
    host_materials.push_back(mat_light_bright);

    add_cube(host_planes, point3(0, 1, 4), 1.f, 2, host_spheres, 1);

    host_planes.push_back({point3(-10, 0, -10), vec3(20, 0, 0), vec3(0, 0, 20), 5, PlaneType::QUAD});
    host_planes.push_back({point3(-100, 25, -100), vec3(200, 0, 0), vec3(0, 0, 200), 3, PlaneType::QUAD});

    // host_spheres.push_back({point3(0, -1000, 0), 1000.0f, 5});

    // host_spheres.push_back({point3(0, 1, 0), 1.0f, 4});
    // host_spheres.push_back({point3(4, 1, 0), 1.0f, 1});
    // host_spheres.push_back({point3(-4, 1, 0), 1.0f, 2});
    // host_spheres.push_back({point3(-0.5, 1.5, 3.5), 0.1f, 6});
    // host_spheres.push_back({point3(0, 1000, 0), 500.0f, 3});
    // host_spheres.push_back({point3(0, 1, -4), 1.0f, 3});

    // host_point_lights.push_back({point3(0, 10, 0), color(100.0, 100.0, 100.0)});
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

int main() {
    const float TOTAL_SWEEP_RAD = 2.0 * M_PI;
    const float ORBIT_RADIUS = 5.0;  // Расстояние от центра
    const float CAMERA_HEIGHT = 1;

    SceneData host_scene;
    std::vector<MaterialData> host_materials;
    std::vector<SphereData> host_spheres;
    std::vector<PlaneData> host_planes;
    std::vector<PointLightData> host_point_lights;

    create_scene(host_materials, host_spheres, host_planes, host_point_lights);  // Заполняем CPU векторы

    host_scene.num_spheres = host_spheres.size();
    host_scene.num_planes = host_planes.size();
    host_scene.num_point_lights = host_point_lights.size();
    host_scene.num_materials = host_materials.size();

    cudaMalloc((void **)&host_scene.d_materials, host_materials.size() * sizeof(MaterialData));
    cudaMemcpy(host_scene.d_materials, host_materials.data(), host_materials.size() * sizeof(MaterialData),
               cudaMemcpyHostToDevice);

    cudaMalloc((void **)&host_scene.d_spheres, host_spheres.size() * sizeof(SphereData));
    cudaMemcpy(host_scene.d_spheres, host_spheres.data(), host_spheres.size() * sizeof(SphereData), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&host_scene.d_planes, host_planes.size() * sizeof(PlaneData));
    cudaMemcpy(host_scene.d_planes, host_planes.data(), host_planes.size() * sizeof(PlaneData), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&host_scene.d_point_lights, host_point_lights.size() * sizeof(PointLightData));
    cudaMemcpy(host_scene.d_point_lights, host_point_lights.data(), host_point_lights.size() * sizeof(PointLightData),
               cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_scene_data_const, &host_scene, sizeof(SceneData), 0, cudaMemcpyHostToDevice);

    color *d_fb = nullptr;
    std::size_t num_pixels = static_cast<std::size_t>(1080) * 16.0 / 9.0 * 1080;
    checkCudaErrors(cudaMalloc(&d_fb, num_pixels * sizeof(color)));

    int frames = 100;
    for (int n = 0; n < frames; ++n) {
        auto saver = std::make_unique<PNGSaver>(100, "../images/render" + std::to_string(n) + ".png");
        float t = float(n) / (frames - 1);
        float angle = t * TOTAL_SWEEP_RAD + M_PI / 4;
        float current_x = ORBIT_RADIUS * std::cos(angle);
        float current_z = ORBIT_RADIUS * std::sin(angle);

        Camera camera(16.0 / 9.0, 1500, std::move(saver), point3(current_x, CAMERA_HEIGHT, current_z), point3(0, 1, 4));
        camera.samplesPerPixel = 1000;
        camera.maxDepth = 50;
        camera.background_color = color(0, 0, 0);

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
