#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
//#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv_modules.hpp>

void gpu_image2Matrix(int width, int height , cv::cuda::GpuMat & image, float* matrix);

