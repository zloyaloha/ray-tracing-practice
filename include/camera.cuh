#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "interval.h"
#include "materials.h"
#include "random_utils.h"
#include "ray.h"
#include "scene.h"
#include "scene_params.h"
#include "vec3.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func
                  << "'\n";
        cudaDeviceReset();
        std::exit(99);
    }
}

class ISaver {
protected:
    int samplesPerPixel;
    int imageWidth;
    int imageHeight;

public:
    explicit ISaver(int samplesPerPixel);
    virtual ~ISaver() = default;
    virtual void writeColor(color pixel_color) = 0;
    float linearToGamma(const float &linear);
    virtual void setFormat(int imageWidth, int imageHeight) = 0;
};

class FileSaver : public ISaver {
public:
    FileSaver(int samplesPerPixel, const std::string &filename);
    void writeColor(color pixel_color) override;
    void setFormat(int imageWidth, int imageHeight) override;

private:
    std::ofstream fout;
};

class PNGSaver : public ISaver {
public:
    PNGSaver(int samplesPerPixel, const std::string &filepath);
    void writeColor(color pixel_color) override;
    void setFormat(int imageWidth, int imageHeight) override;
    ~PNGSaver();

private:
    std::vector<unsigned char> pixel_data;
    std::string filepath;
    int pixel_count = 0;
};

class BinarySaver : public ISaver {
public:
    BinarySaver(int samplesPerPixel, const std::string &filepath);
    void writeColor(color pixel_color) override;
    void setFormat(int imageWidth, int imageHeight) override;
    ~BinarySaver() = default;

private:
    std::ofstream fout;
};

class OutStreamSaver : public ISaver {
public:
    explicit OutStreamSaver(int samplesPerPixel);
    void writeColor(color pixel_color) override;
    void setFormat(int imageWidth, int imageHeight) override;
};

struct CameraData {
    point3 origin;
    point3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    color background;
    int image_width;
    int image_height;
    int samples_per_pixel;
    int max_depth;

    __host__ __device__ Ray get_ray(int i, int j, unsigned int &local_seed) const {
        auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);

        float offset_x = random_float(local_seed) - 0.5f;
        float offset_y = random_float(local_seed) - 0.5f;

        auto pixel_sample = pixel_center + (offset_x * pixel_delta_u) + (offset_y * pixel_delta_v);

        auto ray_origin = origin;
        auto ray_direction = pixel_sample - ray_origin;

        return Ray(ray_origin, ray_direction);
    }
};

__device__ color ray_color(Ray r, unsigned int &local_seed, const SceneData &scene_data_ref, const CameraData &cam_data_ref);
__host__ color ray_color_host(Ray r, unsigned int &local_seed, const SceneData &scene_data_ref, const CameraData &cam_data_ref);
__host__ void gpu_render(SceneParams &scene_params, SceneData &host_scene);
__host__ void cpu_render(SceneParams &scene_params, SceneData &host_scene);

class Camera {
public:
    Camera(int height, int width, std::unique_ptr<ISaver> image_saver, const point3 &camera_pos,
           const point3 &look_at_point = point3(0, 0, 0));

    void render(color *d_fb) const;
    void render_cpu(const SceneData &scene_data, std::vector<color> &framebuffer);
    CameraData build_camera_data() const;

    int imageWidth;
    int imageHeight;
    float aspectRatio;
    int samplesPerPixel;
    int maxDepth;
    color background_color;
    float vfov = 60.0;
    std::unique_ptr<ISaver> saver;
    point3 origin;
    point3 look_at;

private:
    vec3 vup;
};