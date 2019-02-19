#include "inference_base.h"
#include "inference_tf.h"
#include "inference_trt.h"

#include <cuda_profiler_api.h>

using tensorflow::CallableOptions;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;

using namespace std;
using namespace cv;
using namespace std::chrono;

int main(int argc, char *argv[])
{
    if (!tensorflow::IsGoogleCudaEnabled())
    {
        LOG(ERROR) << "Tensorflow built without CUDA. Rebuild with -c opt --config=cuda";
        return -1;
    }

    const String keys =
        "{d display |1  | view video while objects are detected}"
        "{t tensorrt|false | use tensorrt}"
        "{i int8|false| use INT8 (requires callibration)}"
        "{v video    |  | video for detection}"
        "{graph ||frozen graph location}"
        "{labels ||trained labels filelocation}";

    // Set dirs variables
    string ROOTDIR = "";

    CommandLineParser parser(argc, argv, keys);
    int showWindow = parser.get<int>("d");
    String video_file = parser.get<String>("v");
    bool is_tensor_rt = parser.get<bool>("t");
    bool is_int8 = parser.get<bool>("i");
    String LABELS = parser.get<String>("labels");
    String GRAPH = parser.get<String>("graph");

    unique_ptr<InferenceBase> infer(is_tensor_rt ? 
        (InferenceBase *) new InferenceTensorRT(LABELS, GRAPH, is_int8) 
        : (InferenceBase *) new InferenceTensorflow(LABELS, GRAPH));
    
    infer->set_debug(showWindow);
    
    infer->Init(video_file);
    infer->RunInferenceOnStream();
    
    return 0;
}