#include "utils.h"

using namespace std;
using namespace cv;

using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using tensorflow::DeviceAttributes;

/** Read a model graph definition (xxx.pb) from disk, and creates a session object you can use to run it.
 */
Status loadGraph(const string &graph_file_name,
                 unique_ptr<tensorflow::Session> *session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    tensorflow::SessionOptions session_options;
    session_options.config.mutable_gpu_options()->set_allow_growth(true);

    session->reset(tensorflow::NewSession(session_options));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

/** Read a labels map file (xxx.pbtxt) from disk to translate class numbers into human-readable labels.
 */
Status readLabelsMapFile(const string &fileName, map<int, string> &labelsMap) {

    // Read file into a string
    ifstream t(fileName);
    if (t.bad())
        return tensorflow::errors::NotFound("Failed to load labels map at '", fileName, "'");
    stringstream buffer;
    buffer << t.rdbuf();
    string fileString = buffer.str();

    // Search entry patterns of type 'item { ... }' and parse each of them
    smatch matcherEntry;
    smatch matcherId;
    smatch matcherName;
    const regex reEntry("item \\{([\\S\\s]*?)\\}");
    const regex reId("id: [0-9]+");
    const regex reDisplayName("display_name: (\"|\').+(\"|\')");
    const regex reName("name: (\"|\').+(\"|\')");

    string entry;

    const string namePrefix = "name: \"";
    const string display_name = "display_name: \"";

    const size_t idOffset = string("id: ").length();
    size_t nameOffset = display_name.length();

    // we first try to parse "display_name"
    // and fall back if it does not exist
    bool isParsingName = false;

    auto stringBegin = sregex_iterator(fileString.begin(), fileString.end(), reEntry);
    auto stringEnd = sregex_iterator();

    int id;
    string name;
    for (sregex_iterator i = stringBegin; i != stringEnd; i++) {
        matcherEntry = *i;
        entry = matcherEntry.str();
        regex_search(entry, matcherId, reId);
        if (!matcherId.empty())
            id = stoi(matcherId[0].str().substr(idOffset, matcherId[0].str().length() - idOffset));
        else
            continue;

        if(!isParsingName)
        {
            regex_search(entry, matcherName, reDisplayName);
            if(matcherName.empty())
            {
                isParsingName = true;
                nameOffset = namePrefix.length();
            }
        }

        if(isParsingName)
        {
            regex_search(entry, matcherName, reName);
        }
        
        if (!matcherName.empty())
            name = matcherName[0].str().substr(nameOffset, matcherName[0].str().length() - nameOffset - 1);
        else
            continue;
        labelsMap.insert(pair<int, string>(id, name));
    }
    return Status::OK();
}

/** Convert Mat image into tensor of shape (1, height, width, d) where last three dims are equal to the original dims.
 */
Status readTensorFromMat(const Mat &mat, Tensor &outTensor) {
    // Trick from https://github.com/tensorflow/tensorflow/issues/8033
    tensorflow::uint8 *p = outTensor.flat<tensorflow::uint8>().data();
    Mat fakeMat(mat.rows, mat.cols, CV_8UC3, p);
    cv::cvtColor(mat, fakeMat, COLOR_BGR2RGB);
    return Status::OK();
}

Status readTensorFromGpuMat(const cv::cuda::GpuMat& g_mat, Tensor& outTensor) {
    tensorflow::uint8 *p = outTensor.flat<tensorflow::uint8>().data();
    cv::cuda::GpuMat fakeMat(g_mat.rows, g_mat.cols, CV_8UC3, p);
    
    // comes in with 4 channels -> 3 channels
    cv::cuda::cvtColor(g_mat, fakeMat, COLOR_BGRA2RGB);

    return Status::OK();
}

/** Draw bounding box and add caption to the image.
 *  Boolean flag _scaled_ shows if the passed coordinates are in relative units (true by default in tensorflow detection)
 */
void drawBoundingBoxOnImage(Mat &image, double yMin, double xMin, double yMax, double xMax, double score, string label, bool scaled) {
    cv::Point tl, br;
    if (scaled) {
        tl = cv::Point((int) (xMin * image.cols), (int) (yMin * image.rows));
        br = cv::Point((int) (xMax * image.cols), (int) (yMax * image.rows));
    } else {
        tl = cv::Point((int) xMin, (int) yMin);
        br = cv::Point((int) xMax, (int) yMax);
    }
    cv::rectangle(image, tl, br, cv::Scalar(0, 255, 255), 1);

    // Ceiling the score down to 3 decimals (weird!)
    float scoreRounded = floorf(score * 1000) / 1000;
    string scoreString = to_string(scoreRounded).substr(0, 5);
    string caption = label + " (" + scoreString + ")";

    // Adding caption of type "LABEL (X.XXX)" to the top-left corner of the bounding box
    int fontCoeff = 12;
    cv::Point brRect = cv::Point(tl.x + caption.length() * fontCoeff / 1.6, tl.y + fontCoeff);
    cv::rectangle(image, tl, brRect, cv::Scalar(0, 255, 255), -1);
    cv::Point textCorner = cv::Point(tl.x, tl.y + fontCoeff * 0.9);
    cv::putText(image, caption, textCorner, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0));
}

/** Draw bounding boxes and add captions to the image.
 *  Box is drawn only if corresponding score is higher than the _threshold_.
 */
void drawFrameworkSignature(Mat& image, double fps, string signature, Scalar& color)
{
        putText(image, "TensorFlow", Point(0, image.rows - 30), FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        putText(image, to_string(fps).substr(0, 5), Point(0, image.rows - 5), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        imshow("stream", image);
        waitKey(1);
}

void drawBoundingBoxesOnImage(Mat &image,
                              tensorflow::TTypes<float>::Flat &scores,
                              tensorflow::TTypes<float>::Flat &classes,
                              tensorflow::TTypes<float,3>::Tensor &boxes,
                              map<int, string> &labelsMap,
                              vector<size_t> &idxs) {
    for (int j = 0; j < idxs.size(); j++)
        drawBoundingBoxOnImage(image,
                               boxes(0,idxs.at(j),0), boxes(0,idxs.at(j),1),
                               boxes(0,idxs.at(j),2), boxes(0,idxs.at(j),3),
                               scores(idxs.at(j)), labelsMap[classes(idxs.at(j))]);
}

/** Calculate intersection-over-union (IOU) for two given bbox Rects.
 */
double IOU(Rect2f box1, Rect2f box2) {

    float xA = max(box1.tl().x, box2.tl().x);
    float yA = max(box1.tl().y, box2.tl().y);
    float xB = min(box1.br().x, box2.br().x);
    float yB = min(box1.br().y, box2.br().y);

    float intersectArea = abs((xB - xA) * (yB - yA));
    float unionArea = abs(box1.area()) + abs(box2.area()) - intersectArea;

    return 1. * intersectArea / unionArea;
}

/** Return idxs of good boxes (ones with highest confidence score (>= thresholdScore)
 *  and IOU <= thresholdIOU with others).
 */
vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &scores,
                           tensorflow::TTypes<float, 3>::Tensor &boxes,
                           double thresholdIOU, double thresholdScore) {

    vector<size_t> sortIdxs(scores.size());
    iota(sortIdxs.begin(), sortIdxs.end(), 0);

    // Create set of "bad" idxs
    set<size_t> badIdxs = set<size_t>();
    size_t i = 0;
    while (i < sortIdxs.size()) {
        if (scores(sortIdxs.at(i)) < thresholdScore)
            badIdxs.insert(sortIdxs[i]);
        if (badIdxs.find(sortIdxs.at(i)) != badIdxs.end()) {
            i++;
            continue;
        }

        Rect2f box1 = Rect2f(Point2f(boxes(0, sortIdxs.at(i), 1), boxes(0, sortIdxs.at(i), 0)),
                             Point2f(boxes(0, sortIdxs.at(i), 3), boxes(0, sortIdxs.at(i), 2)));
        for (size_t j = i + 1; j < sortIdxs.size(); j++) {
            if (scores(sortIdxs.at(j)) < thresholdScore) {
                badIdxs.insert(sortIdxs[j]);
                continue;
            }
            Rect2f box2 = Rect2f(Point2f(boxes(0, sortIdxs.at(j), 1), boxes(0, sortIdxs.at(j), 0)),
                                 Point2f(boxes(0, sortIdxs.at(j), 3), boxes(0, sortIdxs.at(j), 2)));
            if (IOU(box1, box2) > thresholdIOU)
                badIdxs.insert(sortIdxs[j]);
        }
        i++;
    }

    // Prepare "good" idxs for return
    vector<size_t> goodIdxs = vector<size_t>();
    for (auto it = sortIdxs.begin(); it != sortIdxs.end(); it++)
        if (badIdxs.find(sortIdxs.at(*it)) == badIdxs.end())
            goodIdxs.push_back(*it);

    return goodIdxs;
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

bool IsCUDATensor(const Tensor &t)
{
    cudaPointerAttributes attributes;
    cudaError_t err =
        cudaPointerGetAttributes(&attributes, t.tensor_data().data());
    if (err == cudaErrorInvalidValue)
        return false;
    CHECK_EQ(cudaSuccess, err) << cudaGetErrorString(err);
#if CUDART_VERSION >= 10000    
    return (attributes.type == cudaMemoryTypeDevice);
#else
    return (attributes.memoryType == cudaMemoryTypeDevice);
#endif
}

string GPUDeviceName(Session* session) {
  std::vector<DeviceAttributes> devices;
  TF_CHECK_OK(session->ListDevices(&devices));
  for (const DeviceAttributes& d : devices) {
    LOG(INFO) << "Device: " << d.name();
    if (d.device_type() == "GPU" || d.device_type() == "gpu") {
      return d.name();
    }
  }
  return "";
}