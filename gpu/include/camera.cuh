#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "interval.cuh"
#include "random_utils.cuh"
#include "ray.cuh"
#include "scene.cuh"
#include "vec3.cuh"

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

    // __host__ __device__ CameraData() {};
    // __host__ __device__ ~CameraData() {}

    __device__ vec3 pixel_sample_square(curandState &state) const {
        float dx = curand_uniform_double(&state) - 0.5;
        float dy = curand_uniform_double(&state) - 0.5;
        return dx * pixel_delta_u + dy * pixel_delta_v;
    }

    __device__ Ray get_ray(int i, int j, unsigned int &local_seed) const {
        auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);

        float offset_x = random_float(local_seed) - 0.5f;
        float offset_y = random_float(local_seed) - 0.5f;

        auto pixel_sample = pixel_center + (offset_x * pixel_delta_u) + (offset_y * pixel_delta_v);

        auto ray_origin = origin;
        auto ray_direction = pixel_sample - ray_origin;

        return Ray(ray_origin, ray_direction);
    }
};

extern __constant__ CameraData d_cam_data_const;

class Camera {
public:
    Camera(float ratio, int width, std::unique_ptr<ISaver> image_saver, const point3 &camera_pos,
           const point3 &look_at_point = point3(0, 0, 0));

    void render(SceneData *scene_data, color *d_fb) const;

    int imageWidth;
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
    int imageHeight;

    void build_camera_data(CameraData &data) const;
};
