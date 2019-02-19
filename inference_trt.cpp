#include "inference_trt.h"

using namespace cv;
using namespace std;

int InferenceTensorRT::ReadGraph()
{
    auto runtimeEngineContext = CreateTrtEngineAndContext(graphFile, isInt8);

    runtime = std::get<0>(runtimeEngineContext);
    engine = std::get<1>(runtimeEngineContext);
    context = std::get<2>(runtimeEngineContext);
    return 0;
}

int InferenceTensorRT::ReadClassLabels()
{
    populateClassLabels(labelsVector, labelsFile);
    return 0;
}

int InferenceTensorRT::doInference(cv::cuda::GpuMat &d_frame)
{
    auto inferenceTuple = doInferenceWithTrt(d_frame, context, labelsVector);
    detections = std::get<0>(inferenceTuple);
    numDetections = std::get<1>(inferenceTuple);
    return 0;
}

void InferenceTensorRT::visualize(cv::cuda::GpuMat &d_frame, double fps)
{
    Mat img;
    d_frame.download(img);

    for (int p = 0; p < N; ++p)
    {
        for (int i = 0; i < numDetections[p]; ++i)
        {
            float *det = &detections[0] + (p * detectionOutputParam.keepTopK + i) * 7;
            if (det[2] < visualizeThreshold)
                continue;

            // Output format for each detection is stored in the below order
            // [image_id, label, confidence, xmin, ymin, xmax, ymax]
            assert((int)det[1] < OUTPUT_CLS_SIZE);
            std::string storeName = outFileRoot + labelsVector[(int)det[1]] + "-" + std::to_string(det[2]) + ".jpg";

            if (debug & 0x2)
            {
                // det array idxs: (4, 3) = (y0, x0), (6, 5) = (y1, x1)
                // dets are in absolute coordinates: 0 <= pt <= 1
                drawBoundingBoxOnImage(img, det[4], det[3], det[6], det[5], det[2], labelsVector[(int)det[1]]);
            }
        }
    }

    if (debug & 0x2)
    {
        string framework("TensorRT");
        if (isInt8)
        {
            framework += " (INT8)";
        }

        auto color = Scalar(0, 255, 255);
        drawFrameworkSignature(img, fps, framework, color);
    }
}
