#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "interval.cuh"
#include "materials.cuh"
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

inline __device__ color point_light_contrib(const HitRecord &rec, unsigned int seed) {
    color result(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < d_scene_data_const.num_point_lights; ++i) {
        PointLightData light = d_scene_data_const.d_point_lights[i];

        vec3 light_dir = unit_vector(light.pos - rec.point);
        float dist2 = (light.pos - rec.point).len_squared();
        float NdotL = fmaxf(dot(rec.normal, light_dir), 0.0f);

        Ray shadow_ray(rec.point, light_dir);
        HitRecord tmp;
        if (!hit_scene(shadow_ray, Interval(0.001f, sqrtf(dist2)), tmp)) {
            result += light.emit * NdotL / (4.0f * M_PI * dist2);
        }
    }

    for (int i = 0; i < d_scene_data_const.num_spheres; ++i) {
        SphereData sphere = d_scene_data_const.d_spheres[i];
        MaterialData sphere_mat = d_scene_data_const.d_materials[sphere.material_idx];
        if (sphere_mat.type == DIFFUSE_LIGHT) {
            float u = random_float(seed);  // [0, 1]
            float v = random_float(seed);  // [0, 1]

            float theta = acosf(1 - 2 * u);  // [0, pi]
            float phi = 2.0f * M_PI * v;     // [0, 2*pi]

            float x = sinf(theta) * cosf(phi);
            float y = sinf(theta) * sinf(phi);
            float z = cosf(theta);

            vec3 light_point = sphere.center + sphere.radius * vec3(x, y, z);
            vec3 to_light = light_point - rec.point;
            float dist_sq = to_light.len_squared();
            float dist = sqrtf(dist_sq);
            vec3 light_dir = to_light / dist;

            float NdotL = dot(rec.normal, light_dir);

            if (NdotL > 0.0f) {
                Ray shadow_ray(rec.point + rec.normal * 0.0001f, light_dir);
                HitRecord shadow_rec;

                if (!hit_scene(shadow_ray, Interval(0.001f, dist - 0.001f), shadow_rec)) {
                    float area = 4.0f * M_PI * sphere.radius * sphere.radius;

                    result += sphere_mat.emit * NdotL * area / (dist_sq * 4.0f * M_PI);
                }
            }
        }
    }

    return result;
}

inline __device__ color ray_color(Ray r, unsigned int &local_seed) {
    color final_color(0.0f, 0.0f, 0.0f);
    color cur_attenuation(1.0f, 1.0f, 1.0f);
    Ray cur_ray = r;
    vec3 view_dir = unit_vector(-r.direction());  // Направление от точки к камере/глазу

    for (int depth = 0; depth < d_cam_data_const.max_depth; depth++) {
        HitRecord rec;

        if (hit_scene(cur_ray, Interval(0.001f, 1e30f), rec)) {
            MaterialData material = d_scene_data_const.d_materials[rec.material_idx];

            // 1. Расчет цвета поверхности (Альбедо/Текстура)
            color surface_color = material.albedo;
            if (material.tex_obj != 0) {
                float4 tex_val = tex2D<float4>(material.tex_obj, rec.u, rec.v);
                surface_color = color(tex_val.x, tex_val.y, tex_val.z);
            }

            if (material.type != DIFFUSE_LIGHT) {
                final_color += cur_attenuation * point_light_contrib(rec, local_seed);
            }

            Ray scattered;
            final_color += cur_attenuation * material_emit(material);
            color attenuation;

            if (material_scatter(cur_ray, rec, attenuation, scattered, local_seed, material)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return final_color;
            }
        } else {
            final_color += cur_attenuation * d_cam_data_const.background;
            return final_color;
        }
    }
    return final_color;
}
