#include "cuda_runtime.h"

#include "point_cloud.h"
#include "cu/cudaFilter.h"

namespace prediss_point_cloud {

    // https://github.com/NVIDIA-AI-IOT/cuPCL/tree/main/cuFilter
    void point_cloud::filter(point_cloud& target_pc, const filter_options& options)
    {
        cudaStream_t stream = NULL;
        cudaStreamCreate ( &stream );

        unsigned int nCount = size();
        float *inputData = (float *)points;

        float *outputData = (float *)target_pc.points;
        memset(outputData, 0, sizeof(float)*4*nCount);

        float *input = NULL;
        cudaMallocManaged(&input, sizeof(float) * 4 * nCount, cudaMemAttachHost);
        cudaStreamAttachMemAsync (stream, input );
        cudaMemcpyAsync(input, inputData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        float *output = NULL;
        cudaMallocManaged(&output, sizeof(float) * 4 * nCount, cudaMemAttachHost);
        cudaStreamAttachMemAsync (stream, output );
        cudaStreamSynchronize(stream);

        cudaFilter filterTest(stream);
        FilterParam_t setP;
        FilterType_t type;

        if(filter_options.type == PASSTHROUGH) 
        {
            throw runtime_error("PASSTHROUGH not implemented");

            unsigned int countLeft = 0;
            memset(outputData,0,sizeof(float)*4*nCount);

            FilterType_t type = PASSTHROUGH;
            setP.type = type;
            setP.dim = filter_options.dim;
            setP.upFilterLimits = filter_options.up_filter_limits;
            setP.downFilterLimits = filter_options.down_filter_limits;
            setP.limitsNegative = filter_options.limits_negative;
            filterTest.set(setP);

            cudaDeviceSynchronize();
            filterTest.filter(output, &countLeft, input, nCount);
            checkCudaErrors(cudaMemcpyAsync(outputData, output, sizeof(float) * 4 * countLeft, cudaMemcpyDeviceToHost, stream));
            checkCudaErrors(cudaDeviceSynchronize());

            /*
            Not sure why it uses target point cloud pointer for size.
            Don't know what the check var is doing.
            */
           int check = 0;
            for (size_t i = 0; i < target_pc.size(); ++i) {
                target_pc.add_float_point(create_raw_float_point(
                    output[i*4+0], 
                    output[i*4+1], 
                    output[i*4+2]
                ));
            }

        } 
        else if(filter_options.type == VOXELGRID)
        {
            unsigned int countLeft = 0;
            memset(outputData,0,sizeof(float)*4*nCount);

            type = VOXELGRID;
            setP.type = type;
            setP.voxelX = filter_options.voxel_x;
            setP.voxelY = filter_options.voxel_y;
            setP.voxelZ = filter_options.voxel_z;

            filterTest.set(setP);
            
            int status = 0;
            cudaDeviceSynchronize();
            status = filterTest.filter(output, &countLeft, input, nCount);
            cudaDeviceSynchronize();

            if (status != 0) return;

            /*
            Not sure why it uses target point cloud pointer for size.
            Don't know what the check var is doing.
            */
           int check = 0;
            for (size_t i = 0; i < target_pc.size(); ++i) {
                target_pc.add_float_point(create_raw_float_point(
                    output[i*4+0], 
                    output[i*4+1], 
                    output[i*4+2]
                ));
            }
        }
        else 
        {
            throw runtime_error("Filter type not specified");
        }
        
        cudaFree(input);
        cudaFree(output);
        cudaStreamDestroy(stream);
    } // filter
} // namespace prediss_point_cloud