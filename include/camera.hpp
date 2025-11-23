#pragma once

#include <fstream>
#include <hittable_object.hpp>
#include <ray.hpp>

#include "vec3.hpp"

class ISaver {
protected:
    int sampelsPerPixel;
    int imageWidth;
    int imageHeight;

public:
    ISaver(int sampelsPerPixel);
    virtual ~ISaver() = default;
    virtual void writeColor(color pixel_color) = 0;
    double linearToGamma(const double &linear);
    virtual void setFormat(int imageWidth, int imageHeight) = 0;
};

class FileSaver : public ISaver {
public:
    FileSaver(int sampelsPerPixel, const std::string &filename);
    void writeColor(color pixel_color) override;
    void setFormat(int imageWidth, int imageHeight) override;

private:
    std::ofstream fout;
};

class PNGSaver : public ISaver {
public:
    PNGSaver(int sampelsPerPixel, const std::string &filepath);
    void writeColor(color pixel_color) override;
    void setFormat(int imageWidth, int imageHeight) override;
    ~PNGSaver();

private:
    std::vector<unsigned char> pixel_data;  // RGB буфер
    std::string filepath;
    int pixel_count = 0;
};

class OutStreamSaver : public ISaver {
public:
    OutStreamSaver(int sampelsPerPixel);
    void writeColor(color pixel_color) override;
    void setFormat(int imageWidth, int imageHeight) override;
};

class Camera {
private:
    point3 coords;   // Позиция камеры
    point3 look_at;  // Точка, на которую смотрит

    vec3 forward, right, up;  // Система координат камеры
    vec3 delta_pixel_LR, delta_pixel_UD;
    point3 zeroPixelLoc;
    int imageHeight;
    std::unique_ptr<ISaver> saver;

public:
    int imageWidth = 100;
    double aspectRatio = 1.0;
    int sampelsPerPixel;
    int max_depth;
    color background_color;
    double vfov = 90.0;

    Camera(double ratio, int width, std::unique_ptr<ISaver> &image_saver,
           const point3 &camera_pos,
           const point3 &look_at_point = point3(0, 0, 0));
    void render(const HittableObject &world);
    color rayColor(const Ray &ray, int depth,
                   const HittableObject &world) const;
    static double linearToGamma(const double &linear);

private:
    vec3 pixelSampleSquare() const;
    Ray getRay(int i, int j) const;
};