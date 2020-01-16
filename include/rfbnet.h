#ifndef  RFBNET_H_
#define  RFBNET_H_

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <time.h>
#include <unistd.h>
#include<algorithm>
#include<ctime>
#include <time.h>
#include "kernel.h"
using namespace nvinfer1;

static const int INPUT_H = 300;
static const int INPUT_W = 300;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE =46554 ;

static Logger gLogger;
static int gUseDLACore{-1};
typedef struct Bbox
    {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
        float area;

    }Bbox;

void onnxToTRTModel(const std::string& modelFile, unsigned int maxBatchSize,IHostMemory*& trtModelStream);
void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
void read_txt(vector<float> &data_set,string file);
void softmax(vector<vector<float>> input,vector<vector<float>> &output);
bool cmpScore(Bbox lsh, Bbox rsh);
void nms(vector<Bbox> &boundingBox_, const float overlap_threshold);

#endif
