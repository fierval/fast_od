./build/tf_detector_example -d=$1 \
    -v=/home/boris/Videos/ride_2.mp4 \
    -graph=/home/boris/model/frozen_inference_graph.pb \
    -labels=/home/boris/model/mscoco_label_map.pbtxt