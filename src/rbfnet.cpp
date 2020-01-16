#include "rfbnet.h"

void onnxToTRTModel(const std::string& modelFile, unsigned int maxBatchSize,IHostMemory*& trtModelStream){
    int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;

    IBuilder* builder = createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    auto parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser->parseFromFile(modelFile.c_str(), verbosity))
        {
            string msg("failed to parse onnx file");
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
            exit(EXIT_FAILURE);
        }

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(5 << 20);
    samplesCommon::enableDLA(builder, gUseDLACore);
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    parser->destroy();
    trtModelStream = engine->serialize();
    engine->destroy();
    network->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize){
       const ICudaEngine& engine = context.getEngine();
    assert(engine.getNbBindings() == 2);
    void* buffers[2];
    int inputIndex, outputIndex;
    for (int b = 0; b < engine.getNbBindings(); ++b)
    {
        if (engine.bindingIsInput(b))
            inputIndex = b;
        else
            outputIndex = b;
    }

    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * INPUT_C * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W *INPUT_C* sizeof(float), cudaMemcpyDeviceToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

bool cmpScore(Bbox lsh, Bbox rsh) {
    if (lsh.score < rsh.score)
        return true;
    else
        return false;
}

void nms(vector<Bbox> &boundingBox_, const float overlap_threshold){

    if(boundingBox_.empty()){
        return;
    }
    //对各个候选框根据score的大小进行升序排列
    sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    vector<int> vPick;
    int nPick = 0;
    multimap<float, int> vScores;   //存放升序排列后的score和对应的序号
    const int num_boxes = boundingBox_.size();
    vPick.resize(num_boxes);
    for (int i = 0; i < num_boxes; ++i){
        vScores.insert(pair<float, int>(boundingBox_[i].score, i));
    }
    while(vScores.size() > 0){
        int last = vScores.rbegin()->second;  //反向迭代器，获得vScores序列的最后那个序列号
        vPick[nPick] = last;
        nPick += 1;
        for (multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();){
            int it_idx = it->second;
            maxX = max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
            maxY = max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
            minX = min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
            minY = min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
            //转换成了两个边界框相交区域的边长
            maxX = ((minX-maxX)>0)? (minX-maxX) : 0;
            maxY = ((minY-maxY)>0)? (minY-maxY) : 0;

            IOU = (maxX * maxY)/(boundingBox_.at(it_idx).area + boundingBox_.at(last).area - maxX * maxY);

            if(IOU > overlap_threshold){
                it = vScores.erase(it);    //删除交并比大于阈值的候选框,erase返回删除元素的下一个元素
            }else{
                it++;
            }
        }
    }

    vPick.resize(nPick);
    vector<Bbox> tmp_;
    tmp_.resize(nPick);
    for(int i = 0; i < nPick; i++){
        tmp_[i] = boundingBox_[vPick[i]];
    }
    boundingBox_ = tmp_;
}

void read_txt(vector<float> &data_set,string file){
    ifstream f;
    f.open(file,ios::in);
    float tmp;
    for (int i = 0; i < 7759*4 ; i++)
    {

      f >> tmp;
      data_set.push_back(tmp);
    }


    f.close();
}

void softmax(vector<vector<float>> input,vector<vector<float>> &output){
    vector<float> temp;
    for(int i=0;i<input.size();i++){
        //cout<<i<<endl;
        temp.push_back(exp(input[i][0])/(exp(input[i][0])+exp(input[i][1])));
        temp.push_back(exp(input[i][1])/(exp(input[i][0])+exp(input[i][1])));
        output.push_back(temp);
        temp.clear();
    }
}
