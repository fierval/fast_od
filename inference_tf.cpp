#include "inference_tf.h"

using tensorflow::Status;
using tensorflow::Tensor;
using namespace cv;
using tensorflow::int32;

int InferenceTensorflow::ReadGraph()
{
    LOG(INFO) << "graphFile:" << graphFile;
    Status loadGraphStatus = loadGraph(graphFile, &session);
    if (!loadGraphStatus.ok())
    {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return -1;
    }
    else
        LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;

    return 0;
}

// allocate input tensor
int InferenceTensorflow::Init(string videoStream)
{
    if (InferenceBase::Init(videoStream) != 0)
    {
        return -1;
    }

    // configure callable options
    opts.add_feed(inputLayer);
    for (auto const &value : outputLayer)
    {
        opts.add_fetch(value);
    }

    const string gpu_device_name = GPUDeviceName(session.get());
    opts.clear_fetch_devices();
    opts.mutable_feed_devices()->insert({inputLayer, gpu_device_name});

    auto runStatus = session->MakeCallable(opts, &feed_gpu_fetch_cpu);
    if (!runStatus.ok())
    {
        LOG(ERROR) << "Failed to make callable";
    }

    // allocate tensor on the GPU
    tensorflow::TensorShape shape = tensorflow::TensorShape({1, height, width, 3});

    tensorflow::PlatformGpuId platform_gpu_id(0);

    tensorflow::GPUMemAllocator *sub_allocator =
        new tensorflow::GPUMemAllocator(
            tensorflow::GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
            platform_gpu_id, false /*use_unified_memory*/, {}, {});

    tensorflow::GPUBFCAllocator *allocator =
        new tensorflow::GPUBFCAllocator(sub_allocator, shape.num_elements() * sizeof(tensorflow::uint8), "GPU_0_bfc");

    inputTensor = Tensor(allocator, tensorflow::DT_UINT8, shape);
    
    LOG(INFO) << "Is Cuda Tensor: " << IsCUDATensor(inputTensor);

    return 0;
}

int InferenceTensorflow::doInference(cv::cuda::GpuMat &d_frame)
{
    Status runStatus;
    readTensorFromGpuMat(d_frame, inputTensor);

    runStatus = session->RunCallable(feed_gpu_fetch_cpu, {inputTensor}, &outputs, nullptr);
    if (!runStatus.ok())
    {
        LOG(ERROR) << "Running model failed: " << runStatus;
        return -1;
    }
    return 0;
}

void InferenceTensorflow::visualize(cv::cuda::GpuMat &d_frame, double fps)
{
    // Extract results from the outputs vector
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat numDetections = outputs[3].flat<float>();
    tensorflow::TTypes<float, 3>::Tensor boxes = outputs[0].flat_outer_dims<float, 3>();

    vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, thresholdScore);

    if (debug & 0x1)
    {
        for (size_t i = 0; i < goodIdxs.size(); i++)
            LOG(INFO) << "score:" << scores(goodIdxs.at(i)) << ",class:" << labelsMap[classes(goodIdxs.at(i))]
                      << " (" << classes(goodIdxs.at(i)) << "), box:"
                      << "," << boxes(0, goodIdxs.at(i), 0) << ","
                      << boxes(0, goodIdxs.at(i), 1) << "," << boxes(0, goodIdxs.at(i), 2) << ","
                      << boxes(0, goodIdxs.at(i), 3);
    }
    // Draw bboxes and captions
    if (debug & 0x2)
    {
        Mat frame;
        d_frame.download(frame);

        drawBoundingBoxesOnImage(frame, scores, classes, boxes, labelsMap, goodIdxs);
        auto color = Scalar(255, 0, 255);
        
        drawFrameworkSignature(frame, fps, "Tensorflow", color);
    }

}
