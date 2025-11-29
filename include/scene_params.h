#pragma once

#include <string>
#include <vector>

#include "vec3.h"

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
