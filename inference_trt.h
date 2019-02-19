#pragma once
#include "inference_base.h"

using namespace std;

class InferenceTensorRT : public InferenceBase
{
private:
  IRuntime *runtime;
  ICudaEngine *engine;
  IExecutionContext *context;

  bool isInt8;

  //batch size
  const int N = 1;
  const float visualizeThreshold = 0.5;

  vector<string> labelsVector;
  vector<int> numDetections;
  vector<float> detections;
  string outFileRoot;

protected:
  int ReadGraph() override;
  int ReadClassLabels() override;
  int doInference(cv::cuda::GpuMat &d_frame) override;
  void visualize(cv::cuda::GpuMat&, double) override;

public:
  InferenceTensorRT(const string &labelsFile, const string &graphFile, bool isInt8, double threshScore = 0.5, double threshIOU = 0.8, int dbg = 0, string outFile="")
      : InferenceBase(labelsFile, graphFile, threshScore, threshIOU, dbg)
      , labelsVector()
      , numDetections(N)
      , detections(N * detectionOutputParam.keepTopK * 7)
      , outFileRoot(outFile)
      , isInt8(isInt8)
  {
  }

  virtual ~InferenceTensorRT()
  {
    if(context != nullptr)
    {
      context->destroy();
    }
    if(engine != nullptr)
    {
      engine->destroy();
    }

    if(runtime != nullptr)
    {
      runtime->destroy();
    }
  }
};