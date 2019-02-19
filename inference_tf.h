#pragma once

#include "inference_base.h"

using namespace std;
using tensorflow::CallableOptions;
using tensorflow::Tensor;
using tensorflow::Session;

class InferenceTensorflow : public InferenceBase
{
  private:
    const string inputLayer = "image_tensor:0";
    const vector<string> outputLayer = {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"};

    CallableOptions opts;
    std::unique_ptr<tensorflow::Session> session;
    Session::CallableHandle feed_gpu_fetch_cpu;

    // Allocate input tensor on the gpu
    Tensor inputTensor;
    vector<Tensor> outputs;

  protected:
    int ReadGraph() override;
    int doInference(cv::cuda::GpuMat& d_frame) override;
    void visualize(cv::cuda::GpuMat &d_frame, double) override;

  public:
    InferenceTensorflow(const string &labelsFile, const string &graphFile, double threshScore = 0.5, double threshIOU = 0.8, int dbg = 0) 
    : InferenceBase(labelsFile, graphFile, threshScore, threshIOU, dbg)
    , opts()
    { }

    int Init(string videoStream) override;
    
    virtual ~InferenceTensorflow() {  session->ReleaseCallable(feed_gpu_fetch_cpu);}
};