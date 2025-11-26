#include <curand_kernel.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "camera.cuh"
__constant__ CameraData d_cam_data_const = {};
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "sphere.cuh"
#include "stb_image_write.h"

namespace {

constexpr float pi = 3.1415926535897932385;

__global__ void render_kernel(color *framebuffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= d_cam_data_const.image_width || j >= d_cam_data_const.image_height) return;

    int pixel_index = j * d_cam_data_const.image_width + i;

    color pixel_color(0.0, 0.0, 0.0);
    unsigned int base_pixel_seed = wang_hash((i * d_cam_data_const.image_width) + j);

    for (int s = 0; s < d_cam_data_const.samples_per_pixel; ++s) {
        unsigned int local_rand_seed = wang_hash(base_pixel_seed + s);
        Ray r = d_cam_data_const.get_ray(i, j, local_rand_seed);
        pixel_color += ray_color(r, local_rand_seed);
    }

    framebuffer[pixel_index] = pixel_color;
}
}  // namespace

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
        if (result) {
            std::cout << "PNG saved: " << filepath << std::endl;
        } else {
            std::cerr << "Failed to save PNG: " << filepath << std::endl;
        }
    }
}

Camera::Camera(float ratio, int width, std::unique_ptr<ISaver> image_saver, const point3 &camera_pos, const point3 &look_at_point)
    : imageWidth(width),
      aspectRatio(ratio),
      samplesPerPixel(300),
      maxDepth(50),
      background_color(0.0, 0.0, 0.0),
      origin(camera_pos),
      look_at(look_at_point),
      vup(0, 1, 0),
      imageHeight(0),
      saver(std::move(image_saver)) {
    imageHeight = static_cast<int>(imageWidth / aspectRatio);
    imageHeight = (imageHeight < 1) ? 1 : imageHeight;
    if (saver) {
        saver->setFormat(imageWidth, imageHeight);
    }
}

void Camera::build_camera_data(CameraData &data) const {
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
    // data.background = vec3(0.05, 0.05, 0.05);
    data.background = background_color;
    data.image_width = imageWidth;
    data.image_height = imageHeight;
    data.samples_per_pixel = samplesPerPixel;
    data.max_depth = maxDepth;
}

void Camera::render(SceneData *scene_data, color *d_fb) const {
    CameraData cam_data;
    build_camera_data(cam_data);

    cudaMemcpyToSymbol(d_cam_data_const, &cam_data, sizeof(CameraData), 0, cudaMemcpyHostToDevice);

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