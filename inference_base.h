#pragma once
#include "utils.h"

using namespace std;

class InferenceBase
{
  private:
    bool isCudaInited;

    cv::Ptr<cv::cudacodec::VideoReader> GetVideoReader(string video_file)
     {return cv::cudacodec::createVideoReader(video_file);}

  protected:
    string labelsFile;
    string graphFile;
    map<int, string> labelsMap;

    virtual int ReadClassLabels();
    virtual int ReadGraph() = 0;
    void InitCuda();

    cv::Ptr<cv::cudacodec::VideoReader> d_reader;

    double thresholdScore;
    double thresholdIOU;

    // frame width and height
    int height;
    int width;
    
    int debug;

    bool isInitialized;

  public:
    InferenceBase(const string &labelsFile, const string &graphFile, double threshScore, double threshIOU, int dbg)
        : labelsFile(labelsFile)
        , graphFile(graphFile)
        , isCudaInited(false)
        , thresholdScore(threshScore)
        , thresholdIOU(threshIOU)
        , isInitialized(false)
        , labelsMap()
        , debug(dbg)
        {}
    virtual ~InferenceBase() {}

    void RunInferenceOnStream();

    virtual int doInference(cv::cuda::GpuMat&) = 0;
    virtual void visualize(cv::cuda::GpuMat&, double) = 0;

    virtual int Init(string video_stream);

    map<int, string> get_labels_map() {return labelsMap;}

    void set_debug(int dbg) {debug = dbg;}
};

