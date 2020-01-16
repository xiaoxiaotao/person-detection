#include "kernel.h"

using namespace cv::cuda;


inline __device__ __host__ int iDivUp( int a, int b ) { return (a % b != 0) ? (a / b + 1) : (a / b);}

__global__ void image2Matrix_kernel(int width, int height,  PtrStepSz<uchar3> image, float* matrix)
{


	const int w = blockIdx.x * blockDim.x + threadIdx.x;
	const int h = blockIdx.y * blockDim.y + threadIdx.y;
    
    
    if (w < width && h < height)
    {
        uchar3 v = image(h,w);
        *(matrix + 0*height*width + h*width + w) = float(v.x);
        *(matrix + 1*height*width + h*width + w) = float(v.y);
        *(matrix + 2*height*width + h*width + w) = float(v.z);
    }

}
//, cudaStream_t &stream
void gpu_image2Matrix(int width, int height,  cv::cuda::GpuMat & image, float* matrix)
{
     /*
        image : input image in GpuMat format, WHC arrangement and BGR order
        matrix: gpu float array, CHW and RGB order
    */
    //dim3 block(width, height); // width * height blocks, 1 thread each
    const dim3 blockDim(32, 32);
    const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));
    image2Matrix_kernel<<<gridDim,blockDim>>>(width,height,image,matrix);
}

