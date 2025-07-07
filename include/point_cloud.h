#ifndef POINT_CLOUD_H
#define POINT_CLOUD_H

#include <cstdint>
#include <limits>
#include <string>
#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif


using namespace std;

namespace prediss_point_cloud {
    // Processed type for points in the point cloud
    #if defined(__ARM_NEON__) || defined(__ARM_NEON)
    typedef uint32x4_t raw_uint_point;
    typedef float32x4_t raw_float_point;
    #else 
    typedef array<uint32_t, 3> raw_uint_point;
    typedef array<float, 3> raw_float_point;
    #endif

    // Return type for points in the point cloud
    typedef array<float, 3> point;

    constexpr uint16_t max_points = numeric_limits<uint16_t>::max();
    // Currently 3, can change if aditional data needs to be bundled with the point
    constexpr size_t point_size = 3;

    struct cluster_options {
        size_t min_cluster_size = 100;
        size_t max_cluster_size = 2500000;
        float voxel_x = 0.05;
        float voxel_y = 0.05;
        float voxel_z = 0.05;
        size_t count_threshold = 20;
    };

    struct segment_options {
        float distance_threshold = 0.01; 
        size_t max_iterations = 50;
        float probability = 0.99;
        bool optimize_coefficients = true;
    };

    struct filter_options {
        int type = PASSTHROUGH;
        // parameters for PASSTHROUGH
        size_t dim = 0;
        float up_filter_limits = 0.5;
        float down_filter_limits = -0.5;
        bool limits_negative = false;

        // Additional parameters for VOXELGRID
        size_t voxel_x = 1;
        size_t voxel_y = 1;
        size_t voxel_z = 1;

    };

    raw_uint_point create_raw_uint_point(uint32_t x, uint32_t y, uint32_t z)
    {
        #if defined(__ARM_NEON__) || defined(__ARM_NEON)
        raw_uint_point p = {x, y, z, 0};
        #else
        raw_point p = {x, y, z};
        #endif

        return p;
    }
    raw_float_point create_raw_float_point(float x, float y, float z)
    {
        #if defined(__ARM_NEON__) || defined(__ARM_NEON)
        raw_float_point p = {x, y, z, 0.0f};
        #else
        raw_float_point p = {x, y, z};
        #endif

        return p;
    }

struct point_cloud {
    float points[max_points * point_size];      // Store as float on CPU
    uint64_t timestamp_ns;                      // Timestamp in nanoseconds
    size_t n_points;                            // Number of points in the point cloud

    point_cloud() : timestamp_ns(0), n_points(0) {}

    point get_point(int point_index) const 
    {
        if (n_points == 0) {
            throw out_of_range("No points in the point cloud");
        }

        if (point_index < 0 || point_index >= n_points) {
            throw out_of_range("Point index out of bounds");
        }
        return {points[point_index * 3], points[point_index * 3 + 1], points[point_index * 3 + 2]};
    }

    void set_uint_point(int point_index, raw_uint_point values) 
    {
        if (point_index < 0 || point_index >= max_points - 1 ) {
            throw out_of_range("Point index out of bounds");
        }

        #if defined(__ARM_NEON__) || defined(__ARM_NEON)
        // f32 lane of point data
        float32x4_t float_values = vcvtq_f32_u32(values);
        
        // f32 lane of 1000.0f(divisor)
        float32x4_t divisor = vdupq_n_f32(1000.0f);
        float32x4_t result = vdivq_f32(float_values, divisor);

        // store the results
        vst1q_lane_f32(&points[point_index * 3], result, 0);
        vst1q_lane_f32(&points[point_index * 3 + 1], result, 1);
        vst1q_lane_f32(&points[point_index * 3 + 2], result, 2);
        #else
        points[point_index * 3] = values[0] / 1000.0f;
        points[point_index * 3 + 1] = values[1] / 1000.0f;
        points[point_index * 3 + 2] = values[2] / 1000.0f;
        #endif

        n_points++;
    }

    void add_uint_point(raw_uint_point values) 
    {   
        return set_uint_point(n_points, values);
    }

    void set_float_point(int point_index, raw_float_point values) 
    {
        if (point_index < 0 || point_index >= max_points - 1 ) {
            throw out_of_range("Point index out of bounds");
        }

        #if defined(__ARM_NEON__) || defined(__ARM_NEON)
        // f32 lane of point data
        float32x4_t float_values = values;

        // store the results
        vst1q_lane_f32(&points[point_index * 3], float_values, 0);
        vst1q_lane_f32(&points[point_index * 3 + 1], float_values, 1);
        vst1q_lane_f32(&points[point_index * 3 + 2], float_values, 2);
        #else
        points[point_index * 3] = values[0];
        points[point_index * 3 + 1] = values[1];
        points[point_index * 3 + 2] = values[2];
        #endif

        n_points++;
    }

    void add_float_point(raw_float_point values) 
    {   
        return set_float_point(n_points, values);
    }

    size_t size() const 
    {
        // Each point consists of 3 uint32_t values (x, y, z)
        return n_points * 3;
    }

    size_t len() const 
    {
        return n_points;
    }

    void cluster(point_cloud& target_pc, const cluster_options& options);
    void filter(point_cloud& target_pc, const filter_options& options);
    void segment(point_cloud& target_pc, const segment_options& options);

    /*
     * Other pc methods
     */

};
}

#endif // POINT_CLOUD_H