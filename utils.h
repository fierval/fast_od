#ifndef TF_DETECTOR_EXAMPLE_UTILS_H
#define TF_DETECTOR_EXAMPLE_UTILS_H

#endif //TF_DETECTOR_EXAMPLE_UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include <map>
#include <unordered_map>
#include <math.h>
#include <regex>
#include <tuple>

#include <cassert>
#include <cublas_v2.h>
#include <cudnn.h>
#include <sstream>
#include <time.h>

#include "BatchStreamPPM.h"
#include "NvUffParser.h"
#include "common.h"
#include "NvInferPlugin.h"

// Required for CUDA check
#include "tensorflow/core/util/port.h"

// GPU allocator
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"

// Direct session
#include "tensorflow/core/common_runtime/direct_session.h"

#include <cv.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

// CUDA includes. Order matters
#include <dynlink_nvcuvid.h>
#include "cuda_runtime_api.h"

using namespace std;

using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::Session;

using namespace nvinfer1;
using namespace nvuffparser;

string type2str(int type);

Status readLabelsMapFile(const string &fileName, std::map<int, string> &labelsMap);

Status loadGraph(const string &graph_file_name,
                 std::unique_ptr<tensorflow::Session> *session);

Status readTensorFromMat(const cv::Mat &mat, Tensor &outTensor);
Status readTensorFromGpuMat(const cv::cuda::GpuMat& g_mat, Tensor& outTensor);

void drawBoundingBoxOnImage(cv::Mat &image, double xMin, double yMin, double xMax, double yMax, double score, std::string label, bool scaled = true);

void drawBoundingBoxesOnImage(cv::Mat &image,
                              tensorflow::TTypes<float>::Flat &scores,
                              tensorflow::TTypes<float>::Flat &classes,
                              tensorflow::TTypes<float,3>::Tensor &boxes,
                              std::map<int, string> &labelsMap,
                              std::vector<size_t> &idxs);

void drawFrameworkSignature(cv::Mat& image, double fps, string signature, cv::Scalar& color);

double IOU(cv::Rect box1, cv::Rect box2);

std::vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &scores,
                                tensorflow::TTypes<float, 3>::Tensor &boxes,
                                double thresholdIOU, double thresholdScore);

bool IsCUDATensor(const Tensor &t);

string GPUDeviceName(Session* session);

std::tuple<vector<float>, vector<int>> doInferenceWithTrt(cv::cuda::GpuMat& img, IExecutionContext * context, vector<std::string>& CLASSES);

std::tuple<IRuntime*, ICudaEngine *, IExecutionContext*> CreateTrtEngineAndContext(std::string &graphFileName, bool isInt8);

extern DetectionOutputParameters detectionOutputParam;

void populateClassLabels(std::vector<std::string>& CLASSES, const std::string &labelFileName);

void channelFirst(unsigned char * source, float * dest, int channelSize, int channelsNum, int rowElements, int rowSize);

extern const int OUTPUT_CLS_SIZE;
extern const int OUTPUT_BBOX_SIZE;
