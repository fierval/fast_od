./build/tf_detector_example \
    -d=$1 \
    -i \
    -t \
    -v=/home/boris/Videos/ride_2.mp4 \
    -graph=/usr/src/tensorrt/data/ssd/sample_ssd_relu6.uff \
    -labels=/usr/src/tensorrt/data/ssd/ssd_coco_labels.txt