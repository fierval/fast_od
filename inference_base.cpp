#include "inference_base.h"

using tensorflow::Status;
using namespace std;
using namespace cv;
using namespace std::chrono;

int InferenceBase::ReadClassLabels()
{
    Status readLabelsMapStatus = readLabelsMapFile(labelsFile, labelsMap);
    if (!readLabelsMapStatus.ok())
    {
        LOG(ERROR) << "readLabelsMapFile(): ERROR" << readLabelsMapFile;
        return -1;
    }
    else
        LOG(INFO) << "readLabelsMapFile(): labels map loaded with " << labelsMap.size() << " label(s)" << endl;

    return 0;
}

void InferenceBase::InitCuda()
{
    void *hHandleDriver = nullptr;
    CUresult cuda_res = cuInit(0, __CUDA_API_VERSION, hHandleDriver);
    if (cuda_res != CUDA_SUCCESS)
    {
        throw exception();
    }
    cuda_res = cuvidInit(0);
    if (cuda_res != CUDA_SUCCESS)
    {
        throw exception();
    }
    std::cout << "CUDA init: SUCCESS" << endl;
    cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
    isCudaInited = true;
}

int InferenceBase::Init(string videoStream)
{
    if (!isCudaInited)
    {
        InitCuda();
    }

    if (ReadClassLabels() != 0)
    {
        return -1;
    }

    if (ReadGraph() != 0)
    {
        LOG(ERROR) << "Could not load inference graph";
        return -1;
    }
    LOG(INFO) << "Inference graph loaded";

    // create video stream
    d_reader = GetVideoReader(videoStream);
    if (d_reader == nullptr)
    {
        LOG(ERROR) << "Could not create video stream";
        throw exception();
    }

    // save off frame dimensions
    auto formatStruct = d_reader->format();
    width = formatStruct.width;
    height = formatStruct.height;

    isInitialized = true;
    return 0;
}

void InferenceBase::RunInferenceOnStream()
{
    if (!isInitialized)
    {
        LOG(ERROR) << "Video streaming not initialized";
        return;
    }

    cuda::GpuMat d_frame;
    int iFrame = 0, nFrames = 30;
    double fps = 0., infer_tf_ms = 0.;
    high_resolution_clock::time_point start = high_resolution_clock::now();
    high_resolution_clock::time_point end;
    double duration = 0.;

    for (;;)
    {
        start = high_resolution_clock::now();
        if (!d_reader->nextFrame(d_frame))
        {
            break;
        }

        if (doInference(d_frame) != 0)
        {
            LOG(ERROR) << "Inference failed";
            return;
        }
        end = high_resolution_clock::now();
        duration += (double) duration_cast<milliseconds>(end - start).count();

        visualize(d_frame, fps);

        if (++iFrame % nFrames == 0)
        {
            
            fps = 1. * nFrames / duration * 1000.;
            duration = 0.;
        }

        if (iFrame % 100 == 0)
        {
            LOG(INFO) << "Speed: " << to_string(fps).substr(0, 5);
        }
    }
}