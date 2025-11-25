#include <cmath>
#include <iostream>
#include <memory>

#include "camera.cuh"
#include "material.cuh"
#include "scene.cuh"
#include "sphere.cuh"

void create_scene(std::vector<MaterialData> &host_materials, std::vector<SphereData> &host_spheres) {
    // 0: Материал для земли
    MaterialData mat_ground;
    mat_ground.type = LAMBERTIAN;
    mat_ground.albedo = color(0.8f, 0.8f, 0.0f);
    host_materials.push_back(mat_ground);

    // 1: Металл
    MaterialData mat_metal;
    mat_metal.type = METAL;
    mat_metal.albedo = color(0.8f, 0.8f, 0.8f);
    mat_metal.fuzz = 0.2f;
    host_materials.push_back(mat_metal);

    // 2: Диэлектрик (стекло)
    MaterialData mat_glass;
    mat_glass.type = DIELECTRIC;
    mat_glass.ir = 1.5f;
    host_materials.push_back(mat_glass);

    // 3: Диффузный свет
    MaterialData mat_light;
    mat_light.type = DIFFUSE_LIGHT;
    mat_light.emit = color(3.0f, 3.0f, 3.0f);
    host_materials.push_back(mat_light);

    // 3: Диффузный свет
    MaterialData mat_sphere1;
    mat_sphere1.type = LAMBERTIAN;
    mat_sphere1.albedo = color(0.8f, 0.1f, 0.0f);
    host_materials.push_back(mat_sphere1);

    // Земля (использует material_index = 0)
    host_spheres.push_back({point3(0, -1000, 0), 1000.0f, 0});

    // Сфера 1 (использует material_index = 0)
    host_spheres.push_back({point3(0, 0, 0), 1.0f, 4});

    // Сфера 2 (Металл, использует material_index = 1)
    host_spheres.push_back({point3(4, 1, 0), 1.0f, 1});

    // Сфера 3 (Стекло, использует material_index = 2)
    host_spheres.push_back({point3(-4, 1, 0), 1.0f, 2});

    // Сфера 4 (Свет, использует material_index = 3)
    host_spheres.push_back({point3(5, 5, 5), 1.0f, 3});
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
    const float CAMERA_HEIGHT = 5;

    SceneData host_scene;
    std::vector<MaterialData> host_materials;
    std::vector<SphereData> host_spheres;
    create_scene(host_materials, host_spheres);  // Заполняем CPU векторы

    host_scene.num_spheres = host_spheres.size();
    host_scene.num_materials = host_materials.size();

    cudaMalloc((void **)&host_scene.d_materials, host_materials.size() * sizeof(MaterialData));
    cudaMemcpy(host_scene.d_materials, host_materials.data(), host_materials.size() * sizeof(MaterialData),
               cudaMemcpyHostToDevice);

    cudaMalloc((void **)&host_scene.d_spheres, host_spheres.size() * sizeof(SphereData));
    cudaMemcpy(host_scene.d_spheres, host_spheres.data(), host_spheres.size() * sizeof(SphereData), cudaMemcpyHostToDevice);

    color *d_fb = nullptr;
    std::size_t num_pixels = static_cast<std::size_t>(1080) * 16.0 / 9.0 * 1080;
    checkCudaErrors(cudaMalloc(&d_fb, num_pixels * sizeof(color)));

    int frames = 100;
    for (int n = 0; n < frames; ++n) {
        auto saver = std::make_unique<PNGSaver>(100, "../images/render" + std::to_string(n) + ".png");
        float t = float(n) / (frames - 1);
        float angle = t * TOTAL_SWEEP_RAD;
        float current_x = ORBIT_RADIUS * std::cos(angle);
        float current_z = ORBIT_RADIUS * std::sin(angle);

        Camera camera(16.0 / 9.0, 1080, std::move(saver), point3(current_x, CAMERA_HEIGHT, current_z), point3(0, 0, 0));
        camera.samplesPerPixel = 10000;
        camera.maxDepth = 50;
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
        std::cout << "Frame generated in " << time << " ms" << std::endl;
    }
    destroy_scene_arrays(&host_scene);
    cudaFree(d_fb);

    std::cout << "Rendering completed. Output saved to render_gpu.png\n";
    return 0;
}
