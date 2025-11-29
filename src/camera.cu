#include <curand_kernel.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "camera.cuh"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr float pi = 3.1415926535897932385;

__constant__ CameraData d_cam_data_const;
__constant__ SceneData d_scene_data_const;

__global__ void __launch_bounds__(256, 4) render_kernel(color *framebuffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= d_cam_data_const.image_width || j >= d_cam_data_const.image_height) return;

    int pixel_index = j * d_cam_data_const.image_width + i;

    color pixel_color(0.0, 0.0, 0.0);
    unsigned int base_pixel_seed = wang_hash((i * d_cam_data_const.image_width) + j);

    for (int s = 0; s < d_cam_data_const.samples_per_pixel; ++s) {
        unsigned int local_rand_seed = wang_hash(base_pixel_seed + s);
        Ray r = d_cam_data_const.get_ray(i, j, local_rand_seed);
        pixel_color += ray_color(r, local_rand_seed, d_scene_data_const, d_cam_data_const);
    }

    framebuffer[pixel_index] = pixel_color;
}

__host__ void Camera::render_cpu(const SceneData &scene_data, std::vector<color> &framebuffer) {
    CameraData cam_data = build_camera_data();
    for (size_t j = 0; j < cam_data.image_height; ++j) {
        for (size_t i = 0; i < cam_data.image_width; ++i) {
            color pixel_color(0, 0, 0);
            unsigned int base_pixel_seed = wang_hash((i * cam_data.image_width) + j);
            for (size_t sample = 0; sample < cam_data.samples_per_pixel; ++sample) {
                unsigned int local_rand_seed = wang_hash(base_pixel_seed + sample);
                Ray r = cam_data.get_ray(i, j, local_rand_seed);
                pixel_color += ray_color_host(r, local_rand_seed, scene_data, cam_data);
            }
            framebuffer[j * cam_data.image_width + i] = pixel_color;
        }
    }
}

ISaver::ISaver(int samplesPerPixel) : samplesPerPixel(samplesPerPixel), imageWidth(0), imageHeight(0) {}

float ISaver::linearToGamma(const float &linear) { return std::sqrt(linear); }

FileSaver::FileSaver(int samplesPerPixel, const std::string &filename) : ISaver(samplesPerPixel), fout(filename) {}

void FileSaver::setFormat(int width, int height) {
    imageWidth = width;
    imageHeight = height;
    fout << "P3\n" << imageWidth << ' ' << imageHeight << "\n255\n";
}

void FileSaver::writeColor(color pixel_color) {
    pixel_color = pixel_color / samplesPerPixel;
    float r = linearToGamma(pixel_color.x());
    float g = linearToGamma(pixel_color.y());
    float b = linearToGamma(pixel_color.z());

    static const Interval intensity(0.0, 0.999);
    fout << static_cast<int>(256 * intensity.clamp(r)) << ' ' << static_cast<int>(256 * intensity.clamp(g)) << ' '
         << static_cast<int>(256 * intensity.clamp(b)) << '\n';
}

OutStreamSaver::OutStreamSaver(int samplesPerPixel) : ISaver(samplesPerPixel) {}

void OutStreamSaver::setFormat(int width, int height) {
    imageWidth = width;
    imageHeight = height;
    std::cout << "P3\n" << imageWidth << ' ' << imageHeight << "\n255\n";
}

void OutStreamSaver::writeColor(color pixel_color) {
    pixel_color = pixel_color / samplesPerPixel;
    float r = linearToGamma(pixel_color.x());
    float g = linearToGamma(pixel_color.y());
    float b = linearToGamma(pixel_color.z());

    static const Interval intensity(0.0, 0.999);
    std::cout << static_cast<int>(256 * intensity.clamp(r)) << ' ' << static_cast<int>(256 * intensity.clamp(g)) << ' '
              << static_cast<int>(256 * intensity.clamp(b)) << '\n';
}

PNGSaver::PNGSaver(int samplesPerPixel, const std::string &filepath) : ISaver(samplesPerPixel), filepath(filepath) {}

void PNGSaver::setFormat(int width, int height) {
    imageWidth = width;
    imageHeight = height;
    pixel_data.resize(static_cast<std::size_t>(width) * height * 3, 0);
    pixel_count = 0;
}

void PNGSaver::writeColor(color pixel_color) {
    pixel_color = pixel_color / samplesPerPixel;
    float r = linearToGamma(pixel_color.x());
    float g = linearToGamma(pixel_color.y());
    float b = linearToGamma(pixel_color.z());

    static const Interval intensity(0.0, 0.999);

    unsigned char r_byte = static_cast<unsigned char>(256 * intensity.clamp(r));
    unsigned char g_byte = static_cast<unsigned char>(256 * intensity.clamp(g));
    unsigned char b_byte = static_cast<unsigned char>(256 * intensity.clamp(b));

    int idx = pixel_count * 3;
    pixel_data[idx] = r_byte;
    pixel_data[idx + 1] = g_byte;
    pixel_data[idx + 2] = b_byte;
    pixel_count++;
}

PNGSaver::~PNGSaver() {
    if (!pixel_data.empty()) {
        int result = stbi_write_png(filepath.c_str(), imageWidth, imageHeight, 3, pixel_data.data(), imageWidth * 3);
    }
}

BinarySaver::BinarySaver(int samplesPerPixel, const std::string &filepath)
    : ISaver(samplesPerPixel), fout(filepath, std::ios::binary) {}

void BinarySaver::setFormat(int width, int height) {
    imageWidth = width;
    imageHeight = height;
    fout.write(reinterpret_cast<const char *>(&imageWidth), sizeof(int));
    fout.write(reinterpret_cast<const char *>(&imageHeight), sizeof(int));
}

void BinarySaver::writeColor(color pixel_color) {
    pixel_color = pixel_color / samplesPerPixel;
    float r = linearToGamma(pixel_color.x());
    float g = linearToGamma(pixel_color.y());
    float b = linearToGamma(pixel_color.z());

    static const Interval intensity(0.0, 0.999);

    unsigned char r_byte = static_cast<unsigned char>(256 * intensity.clamp(r));
    unsigned char g_byte = static_cast<unsigned char>(256 * intensity.clamp(g));
    unsigned char b_byte = static_cast<unsigned char>(256 * intensity.clamp(b));

    fout.write(reinterpret_cast<const char *>(&r_byte), sizeof(unsigned char));
    fout.write(reinterpret_cast<const char *>(&g_byte), sizeof(unsigned char));
    fout.write(reinterpret_cast<const char *>(&b_byte), sizeof(unsigned char));
}

Camera::Camera(int height, int width, std::unique_ptr<ISaver> image_saver, const point3 &camera_pos, const point3 &look_at_point)
    : imageWidth(width),
      imageHeight(height),
      aspectRatio((float)width / height),
      samplesPerPixel(300),
      maxDepth(50),
      background_color(0.0, 0.0, 0.0),
      origin(camera_pos),
      look_at(look_at_point),
      vup(0, 0, 1),
      saver(std::move(image_saver)) {
    if (saver) {
        saver->setFormat(imageWidth, imageHeight);
    }
}

CameraData Camera::build_camera_data() const {
    CameraData data;
    float theta = vfov * pi / 180.0;
    float h = tan(theta / 2);
    float viewport_height = 2.0 * h;
    float viewport_width = viewport_height * (static_cast<float>(imageWidth) / imageHeight);

    vec3 w = unit_vector(origin - look_at);
    vec3 u = unit_vector(cross(vup, w));
    vec3 v = cross(w, u);

    vec3 horizontal = viewport_width * u;
    vec3 vertical = viewport_height * v;

    data.origin = origin;
    data.pixel_delta_u = horizontal / imageWidth;
    data.pixel_delta_v = -vertical / imageHeight;
    point3 upper_left = origin - w - horizontal / 2 + vertical / 2;
    data.pixel00_loc = upper_left + 0.5 * (data.pixel_delta_u + data.pixel_delta_v);
    data.background = background_color;
    data.image_width = imageWidth;
    data.image_height = imageHeight;
    data.samples_per_pixel = samplesPerPixel;
    data.max_depth = maxDepth;
    return data;
}

void Camera::render(color *d_fb) const {
    std::size_t num_pixels = static_cast<std::size_t>(imageWidth) * imageHeight;

    dim3 threads(16, 16);
    dim3 blocks((imageWidth + threads.x - 1) / threads.x, (imageHeight + threads.y - 1) / threads.y);

    render_kernel<<<blocks, threads>>>(d_fb);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<color> host_fb(num_pixels);
    checkCudaErrors(cudaMemcpy(host_fb.data(), d_fb, num_pixels * sizeof(color), cudaMemcpyDeviceToHost));

    for (int j = 0; j < imageHeight; ++j) {
        for (int i = 0; i < imageWidth; ++i) {
            saver->writeColor(host_fb[static_cast<std::size_t>(j) * imageWidth + i]);
        }
    }
}

__device__ color ray_color(Ray r, unsigned int &local_seed, const SceneData &scene_data_ref, const CameraData &cam_data_ref) {
    color final_color(0.0f, 0.0f, 0.0f);
    color beta(1.0f, 1.0f, 1.0f);
    Ray cur_ray = r;

    for (int depth = 0; depth < cam_data_ref.max_depth; depth++) {
        HitRecord rec;

        if (!hit_scene(cur_ray, Interval(0.001f, 1e30f), rec, scene_data_ref)) {
            final_color += beta * cam_data_ref.background;
            break;
        }

        MaterialData material = scene_data_ref.d_materials[rec.material_idx];

        if (material.tex_obj != 0) {
            float4 tex_val = tex2D<float4>(material.tex_obj, rec.u, rec.v);
            material.albedo = color(tex_val.x, tex_val.y, tex_val.z) * material.albedo;
        }
        color emitted = material_emit(material);
        final_color += beta * emitted;

        Ray scattered;
        color attenuation;
        if (!material_scatter(cur_ray, rec, attenuation, scattered, local_seed, material)) {
            break;
        }

        beta *= attenuation;

        cur_ray = scattered;
    }

    return final_color;
}

__host__ color ray_color_host(Ray r, unsigned int &local_seed, const SceneData &scene_data_ref, const CameraData &cam_data_ref) {
    color final_color(0.0f, 0.0f, 0.0f);
    color beta(1.0f, 1.0f, 1.0f);
    Ray cur_ray = r;

    for (int depth = 0; depth < cam_data_ref.max_depth; depth++) {
        HitRecord rec;

        if (!hit_scene(cur_ray, Interval(0.001f, 1e30f), rec, scene_data_ref)) {
            final_color += beta * cam_data_ref.background;
            break;
        }

        MaterialData material = scene_data_ref.d_materials[rec.material_idx];

        if (material.cpu_tex != nullptr) {
            material.albedo = material.albedo * tex2D_cpu(material.cpu_tex, rec.u, rec.v);
        }

        color emitted = material_emit(material);
        final_color += beta * emitted;

        Ray scattered;
        color attenuation;
        if (!material_scatter(cur_ray, rec, attenuation, scattered, local_seed, material)) {
            break;
        }

        beta *= attenuation;

        cur_ray = scattered;
    }

    return final_color;
}

void gpu_render(SceneParams &scene_params, SceneData &host_scene) {
    cudaMemcpyToSymbol(d_scene_data_const, &host_scene, sizeof(SceneData), 0, cudaMemcpyHostToDevice);

    color *d_fb = nullptr;
    std::size_t num_pixels = static_cast<std::size_t>(scene_params.width) * scene_params.height;
    checkCudaErrors(cudaMalloc(&d_fb, num_pixels * sizeof(color)));

    for (int n = 0; n < scene_params.num_frames; ++n) {
        char filename[256];
        snprintf(filename, sizeof(filename), scene_params.output_path.c_str(), n);
        auto saver = std::make_unique<BinarySaver>(scene_params.render.sqrt_rays_per_pixel, filename);
        float t_param = (float(n) / scene_params.num_frames) * 2.0f * M_PI;

        float r_c = scene_params.camera_path.rc0 +
                    scene_params.camera_path.Arc * sin(scene_params.camera_path.wrc * t_param + scene_params.camera_path.prc);
        float z_c = scene_params.camera_path.zc0 +
                    scene_params.camera_path.Azc * sin(scene_params.camera_path.wzc * t_param + scene_params.camera_path.pzc);
        float phi_c = scene_params.camera_path.phic0 + scene_params.camera_path.wc * t_param;
        point3 lookfrom(r_c * cos(phi_c), r_c * sin(phi_c), z_c);

        float r_n = scene_params.camera_path.rn0 +
                    scene_params.camera_path.Arn * sin(scene_params.camera_path.wrn * t_param + scene_params.camera_path.prn);
        float z_n = scene_params.camera_path.zn0 +
                    scene_params.camera_path.Azn * sin(scene_params.camera_path.wzn * t_param + scene_params.camera_path.pzn);
        float phi_n = scene_params.camera_path.phin0 + scene_params.camera_path.wn * t_param;
        point3 lookat(r_n * cos(phi_n), r_n * sin(phi_n), z_n);

        Camera camera(scene_params.height, scene_params.width, std::move(saver), lookfrom, lookat);
        camera.vfov = scene_params.fov_degrees;
        int sqrt_spp = scene_params.render.sqrt_rays_per_pixel;
        camera.samplesPerPixel = sqrt_spp * sqrt_spp;
        camera.maxDepth = scene_params.render.max_depth;
        camera.background_color = color(0, 0, 0);

        CameraData cam_data = camera.build_camera_data();
        cudaMemcpyToSymbol(d_cam_data_const, &cam_data, sizeof(CameraData), 0, cudaMemcpyHostToDevice);

        const CameraData *d_cam_data_ptr;
        checkCudaErrors(cudaGetSymbolAddress((void **)&d_cam_data_ptr, d_cam_data_const));

        const SceneData *d_scene_data_ptr;
        checkCudaErrors(cudaGetSymbolAddress((void **)&d_scene_data_ptr, d_scene_data_const));

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        checkCudaErrors(cudaEventRecord(start));
        camera.render(d_fb);
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float time;
        checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
        checkCudaErrors(cudaEventDestroy(start));
        checkCudaErrors(cudaEventDestroy(stop));
        long long total_rays = (long long)scene_params.width * scene_params.height * scene_params.render.sqrt_rays_per_pixel *
                               scene_params.render.sqrt_rays_per_pixel;
        std::cout << n << "\t" << time << "\t" << total_rays << "\n";
    }
    cudaFree(d_fb);
}

void cpu_render(SceneParams &scene_params, SceneData &host_scene) {
    std::vector<color> framebuffer(scene_params.width * scene_params.height);

    for (int n = 0; n < scene_params.num_frames; ++n) {
        char filename[256];
        snprintf(filename, sizeof(filename), scene_params.output_path.c_str(), n);
        auto saver = std::make_unique<BinarySaver>(scene_params.render.sqrt_rays_per_pixel, filename);
        float t_param = (float(n) / scene_params.num_frames) * 2.0f * M_PI;

        float r_c = scene_params.camera_path.rc0 +
                    scene_params.camera_path.Arc * sin(scene_params.camera_path.wrc * t_param + scene_params.camera_path.prc);
        float z_c = scene_params.camera_path.zc0 +
                    scene_params.camera_path.Azc * sin(scene_params.camera_path.wzc * t_param + scene_params.camera_path.pzc);
        float phi_c = scene_params.camera_path.phic0 + scene_params.camera_path.wc * t_param;
        point3 lookfrom(r_c * cos(phi_c), r_c * sin(phi_c), z_c);

        float r_n = scene_params.camera_path.rn0 +
                    scene_params.camera_path.Arn * sin(scene_params.camera_path.wrn * t_param + scene_params.camera_path.prn);
        float z_n = scene_params.camera_path.zn0 +
                    scene_params.camera_path.Azn * sin(scene_params.camera_path.wzn * t_param + scene_params.camera_path.pzn);
        float phi_n = scene_params.camera_path.phin0 + scene_params.camera_path.wn * t_param;
        point3 lookat(r_n * cos(phi_n), r_n * sin(phi_n), z_n);

        Camera camera(scene_params.height, scene_params.width, std::move(saver), lookfrom, lookat);
        camera.vfov = scene_params.fov_degrees;
        int sqrt_spp = scene_params.render.sqrt_rays_per_pixel;
        camera.samplesPerPixel = sqrt_spp * sqrt_spp;
        camera.maxDepth = scene_params.render.max_depth;
        camera.background_color = color(0, 0, 0);

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        camera.render_cpu(host_scene, framebuffer);
        std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        long long total_rays = (long long)scene_params.width * scene_params.height * scene_params.render.sqrt_rays_per_pixel *
                               scene_params.render.sqrt_rays_per_pixel;
        std::cout << n << "\t" << duration_ms.count() << "\t" << total_rays << std::endl;
        for (size_t j = 0; j < camera.imageHeight; ++j) {
            for (size_t i = 0; i < camera.imageWidth; ++i) {
                camera.saver->writeColor(framebuffer[j * camera.imageWidth + i]);
            }
        }
    }
}