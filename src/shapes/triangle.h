#pragma once

#include "base/shape.h"
#include "euclidean_space/transform.h"

class TriangleMesh {
    public:
        const int *vertex_indices;
        const int triangle_num;
        const Point3f *points;
        const int points_num;

        PBRT_CPU_GPU TriangleMesh(const Transform &transform, const int *_vertex_indices,
                                  int _indices_num, const Point3f *_points, int _points_num)
            : triangle_num(_indices_num / 3), points_num(_points_num) {
            int *temp_vertex_indices = new int[_indices_num];
            memcpy(temp_vertex_indices, _vertex_indices, sizeof(int) * _indices_num);
            vertex_indices = temp_vertex_indices;

            auto temp_points = new Point3f[_points_num];
            for (int i = 0; i < _points_num; i++) {
                temp_points[i] = transform(_points[i]);
            }

            points = temp_points;
        }

        PBRT_CPU_GPU ~TriangleMesh() {
            delete[] vertex_indices;
            delete[] points;
        }
};

class Triangle final : public Shape {
    public:
        const TriangleMesh *mesh;
        const int idx;

        PBRT_CPU_GPU Triangle(int _idx, const TriangleMesh *_mesh) : idx(_idx), mesh(_mesh) {}

        PBRT_GPU ~Triangle() {
            if (idx == 0) {
                delete mesh;
            }
        }

        PBRT_GPU bool intersect(Intersection &intersection, const Ray &ray, double t_min,
                                double t_max) const override {
            // TODO:progress 2024/01/16 recovering

            // printf("triangle_%d intesecting\n", idx);

            return false;
        }

        PBRT_GPU const Material *get_material_ptr() const override {
            return nullptr;
        }
};