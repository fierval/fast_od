# Optimizied Video Object Detection

The completed application runs any [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.mdTensorflow) style object detector in Tensorflow mode (default) and an Inception V2 SSD detector converted from Tensorflow graph to UFF format recognized by TensorRT in TensorRT mode (-t).

## Setting up the environment

Read these [series of posts](https://viralfsharp.com/2019/03/25/supercharging-object-detection-in-video-from-glacial-to-lightning-speed/)

## Building the app

* Clone the [repo](https://github.com/fierval/fast_od).
* Get the frozen graph and the class labels files for Tensorflow from [here](https://github.com/fierval/tensorflow-object-detection-cpp/tree/master/demo/ssd_inception_v2)
* Get the [frozen graph for TensorRT](https://www.dropbox.com/s/nc3tzm95ip356i5/sample_ssd_relu6.uff?dl=0). The class labels file should be available in `/usr/src/tensorrt/data/ssd` directory.
* Build:
```sh
mkdir build
cd build
cmake .. # cmake -DCMAKE_BUILD_TYPE=Debug
```

## Running

Command line options are described in [`main.cpp`](https://github.com/fierval/fast_od/blob/master/main.cpp">):

```cpp
const String keys =
    "{d display |1  | view video while objects are detected}"
    "{t tensorrt|false | use tensorrt}"
    "{i int8|false| use INT8 (requires callibration)}"
    "{v video    |  | video for detection}"
    "{graph ||frozen graph location}"
    "{labels ||trained labels filelocation}";
```

Examples are in `run_*.sh` files in the sources directory. Worth mentioning:

```
-d=0 - run without UX, print out framerate only. -d=2 run with UX
-t - TensorRT graph
-t -i - TensorRT graph with INT8 precision.
```

## Slowdown due to UX
The application uses a bare-bones OpenCV UI for visual feedback (`imshow`) and that causes a significant perf hit, so to measure actual performance we run with `-d=0` which suppresses the UI.
