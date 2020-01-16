#include "rfbnet.h"


int main(int argc, char** argv)
{


    gUseDLACore = samplesCommon::parseDLA(argc, argv);
    IHostMemory* trtModelStream{nullptr};
    onnxToTRTModel("../rfbnet_v2.onnx", 1, trtModelStream);
    assert(trtModelStream != nullptr);

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    if (gUseDLACore >= 0)
    {
        runtime->setDLACore(gUseDLACore);
    }
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
/*********************************************************************************************************************************
 * loc_preds_array:网络输出的box位置
 * conf_preds_array:网络输出的人的置信度
 * boxes_array: prior box的数据
 * variances:方差
 * boxes_decode:中间变量
 * boxes_decode_:网络解码后的结果，包括对prior box和网络输出的处理
 * idx:根据conf_preds_array判断box行人的概率>0.5的id
 * conf:box为行人概率>0.5 网络置信度的输出
 * loc:box为行人概率>0.5 网络位置的输出
 * conf_output:softmax 之后的概率
 * nms_data: nms后的box
 * conf_temp,loc_temp:中间变量
 * 
 * ******************************************************************************************************************************/
    float loc_preds_array[7759][4];
    float conf_preds_array[7759][2];
    float boxes_array[7759][4];
    float variances[2]={0.1,0.2};
    float boxes_decode[7759][4];
    float boxes_decode_[7759][4];
    vector<vector<float>> conf;
    vector<vector<float>> loc;
    vector<vector<float>> conf_output;
    vector<int> idx;
    vector<float> conf_temp;
    vector<float> loc_temp;
    Bbox bbox;
    vector<Bbox> nms_data;
    vector<float> boxes_v1;


    cv::Mat inputimage ;
    cv::Mat inputBlob;
    float output_data[OUTPUT_SIZE];
    std::vector<float> prior_boxes;
    std::string prior_box_file="../boxes.txt";
    float* matrix;
    cv::cuda::GpuMat image_cuda;
    CHECK(cudaMalloc((void**)&matrix,sizeof(float)*INPUT_C * INPUT_H * INPUT_W));
    read_txt(boxes_v1,prior_box_file);//read prior box

    for(int i=0;i<7759;i++){
        boxes_array[i][0]=boxes_v1[i*4+0];
        boxes_array[i][1]=boxes_v1[i*4+1];
        boxes_array[i][2]=boxes_v1[i*4+2];
        boxes_array[i][3]=boxes_v1[i*4+3];
    }
    //inputimage=cv::imread("/home/tao/Pictures/vlcsnap-2020-01-03-16h05m53s741.png");
    cv::VideoCapture cap;
    cap.open("/home/tao/object_tracker/person_track/tensorrt_cuda/images/test.mp4");
    while(true){

        cap.read(inputimage);
        if(inputimage.empty())
            {
                cout<<"cant open the camera"<<endl;
                cap.open("/home/tao/object_tracker/person_track/tensorrt_cuda/images/test.mp4");//读入视频
                usleep(50*1000);
                continue;
            }
        //CPU data
        // float *data = new float[3 * 300 * 300];
        // for (int i = 0; i < inputimage.rows; ++i)
        // {
        //     for (int j = 0; j < inputimage.cols; ++j)
        //     {
        //         data[0*INPUT_H * INPUT_W + i * INPUT_H + j] = static_cast<float>(inputimage.at<cv::Vec3b>(i,j)[0]);
        //         data[1*INPUT_H * INPUT_W + i * INPUT_H + j] = static_cast<float>(inputimage.at<cv::Vec3b>(i,j)[1]);
        //         data[2*INPUT_H * INPUT_W + i * INPUT_H + j] = static_cast<float>(inputimage.at<cv::Vec3b>(i,j)[2]);
        //     }
        // }
        else{
        double t = (double)cv::getTickCount();
        cv::resize(inputimage,inputimage,cv::Size(300,300));
        int w=inputimage.cols;
        int h=inputimage.rows;
        image_cuda.upload(inputimage);
        gpu_image2Matrix(INPUT_H ,INPUT_W ,image_cuda,matrix);//cuda 的opencv读入数据，然后写个接口送入网络
        doInference(*context, matrix, output_data, 1);//do inference
        double t1 = (double)cv::getTickCount();

        //将网络的输出reshape程 7759*4，7759*2
        for (int i = 0; i < 7759; i++){
            loc_preds_array[i][0]=output_data[i*4+0];
            loc_preds_array[i][1]=output_data[i*4+1];
            loc_preds_array[i][2]=output_data[i*4+2];
            loc_preds_array[i][3]=output_data[i*4+3];
            conf_preds_array[i][0] = output_data[7759*4+i*2+0];
            conf_preds_array[i][1] = output_data[7759*4+i*2+1];

            boxes_decode[i][0] = loc_preds_array[i][0] * variances[0] * boxes_array[i][2] + boxes_array[i][0];
            boxes_decode[i][1] = loc_preds_array[i][1] * variances[0] * boxes_array[i][3] + boxes_array[i][1];
            boxes_decode[i][2] = exp(loc_preds_array[i][2] * variances[1]) * boxes_array[i][2];
            boxes_decode[i][3] = exp(loc_preds_array[i][3] * variances[1]) * boxes_array[i][3];

            boxes_decode_[i][0] = boxes_decode[i][0] - boxes_decode[i][2]/2.;
            boxes_decode_[i][1] = boxes_decode[i][1] - boxes_decode[i][3]/2.;
            boxes_decode_[i][2] = boxes_decode[i][0] + boxes_decode[i][2]/2.;
            boxes_decode_[i][3] = boxes_decode[i][1] + boxes_decode[i][3]/2.;

        }
        //选出行人概率大于背景的id
        for(int i=0;i<7759;i++){

            if(output_data[7759*4+i*2+1] > output_data[7759*4+i*2+0])
            {
                idx.push_back(i);
            }
        }

        for(int i=0;i<idx.size();i++){
            int id=idx[i];
            loc_temp.push_back(boxes_decode_[id][0]);
            loc_temp.push_back(boxes_decode_[id][1]);
            loc_temp.push_back(boxes_decode_[id][2]);
            loc_temp.push_back(boxes_decode_[id][3]);
            loc.push_back(loc_temp);
            conf_temp.push_back(conf_preds_array[id][0]);
            conf_temp.push_back(conf_preds_array[id][1]);
            conf.push_back(conf_temp);
            loc_temp.clear();
            conf_temp.clear();
        }
        //softmax计算概率
        softmax(conf,conf_output);
        //将box conf送入Bbox结构体，以便于nms
        for(int i=0;i<idx.size();i++)
        {
            bbox.x1=float(loc[i][0]);
            bbox.y1=float(loc[i][1]);
            bbox.x2=float(loc[i][2]);
            bbox.y2=float(loc[i][3]);
            bbox.score=conf_output[i][1];
            bbox.area=(bbox.x2-bbox.x1)*(bbox.y2-bbox.y1);
            nms_data.push_back(bbox);
        }//nms数据处理
        nms(nms_data,0.35);
        for(int i=0;i<nms_data.size();i++){
            if(nms_data[i].score>0.85){
                //std::cout<<nms_data[i].score<<std::endl;
                int x1=int(nms_data[i].x1*w);
                int y1=int(nms_data[i].y1*w-(w-h)/2);
                int x2=int(nms_data[i].x2*w);
                int y2=int(nms_data[i].y2*w-(w-h)/2);
                //std::cout<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<std::endl;
                cv::rectangle(inputimage,cv::Rect(x1,y1,x2-x1,y2-y1),cv::Scalar(0,255,0),4);
            }
        }

        idx.clear();
        loc.clear();
        conf_output.clear();
        conf.clear();
        nms_data.clear();
        cv::imshow("results",inputimage);  
        cv::waitKey(1);

        t1 = (double)cv::getTickCount() - t1;
        cout<<"decode time: "<<t1*1000. / cv::getTickFrequency()<<endl;

        t = (double)cv::getTickCount() - t;
        cout<<"inference time: "<<t*1000. / cv::getTickFrequency()<<endl;

        }

    }
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
