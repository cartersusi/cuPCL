#include "cuda_runtime.h"

#include "point_cloud.h"
#include "cu/cudaCluster.h"

namespace prediss_point_cloud {
    
    // https://github.com/NVIDIA-AI-IOT/cuPCL/tree/main/cuCluster
    void point_cloud::cluster(point_cloud& target_pc, const cluster_options& options)
    {
        throw runtime_error("Clustering not implemented");

        cudaStream_t stream = NULL;
        cudaStreamCreate (&stream);

        /*add cuda cluster*/
        float *inputEC = NULL;
        unsigned int sizeEC = size();
        cudaMallocManaged(&inputEC, sizeof(float) * 4 * sizeEC, cudaMemAttachHost);
        cudaStreamAttachMemAsync (stream, inputEC);
        cudaMemcpyAsync(inputEC, points, sizeof(float) * 4 * sizeEC, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        float *outputEC = NULL;
        cudaMallocManaged(&outputEC, sizeof(float) * 4 * sizeEC, cudaMemAttachHost);
        cudaStreamAttachMemAsync (stream, outputEC);
        cudaMemcpyAsync(outputEC, points, sizeof(float) * 4 * sizeEC, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        unsigned int *indexEC = NULL;
        cudaMallocManaged(&indexEC, sizeof(float) * 4 * sizeEC, cudaMemAttachHost);
        cudaStreamAttachMemAsync (stream, indexEC);
        cudaMemsetAsync(indexEC, 0, sizeof(float) * 4 * sizeEC, stream);
        cudaStreamSynchronize(stream);

        extractClusterParam_t ecp;
        ecp.minClusterSize = cluster_options.min_cluster_size;
        ecp.maxClusterSize = cluster_options.max_cluster_size;
        ecp.voxelX = cluster_options.voxel_x;
        ecp.voxelY = cluster_options.voxel_y;
        ecp.voxelZ = cluster_options.voxel_z;
        ecp.countThreshold = cluster_options.count_threshold;
        cudaExtractCluster cudaec(stream);
        cudaec.set(ecp);

        cudaec.extract(inputEC, sizeEC, outputEC, indexEC);
        cudaStreamSynchronize(stream);

        int j = 0;
        for (int i = 1; i <= indexEC[0]; i++) {

            unsigned int outoff = 0;
            for (int w = 1; w < i; w++) {
                if (i>1) {
                    outoff += indexEC[w];
                }
            }

            for (size_t k = 0; k < indexEC[i]; ++k) {
                target_pc.add_float_point(create_raw_float_point(
                    outputEC[(outoff+k)*4+0], 
                    outputEC[(outoff+k)*4+1], 
                    outputEC[(outoff+k)*4+2]
                ));
            }
        }

        cudaFree(inputEC);
        cudaFree(outputEC);
        cudaFree(indexEC);
    } // cluster
} // namespace prediss_point_cloud
