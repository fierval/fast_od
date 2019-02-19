cd ~/git/tensorflow
sudo mkdir /usr/local/tensorflow
sudo mkdir /usr/local/tensorflow/include

sudo cp -r tensorflow/contrib/makefile/downloads/eigen/Eigen /usr/local/tensorflow/include/
sudo cp -r tensorflow/contrib/makefile/downloads/eigen/unsupported /usr/local/tensorflow/include/

sudo cp tensorflow/contrib/makefile/downloads/nsync/public/* /usr/local/tensorflow/include/
sudo cp -r bazel-genfiles/tensorflow /usr/local/tensorflow/include/
sudo cp -r tensorflow/cc /usr/local/tensorflow/include/tensorflow
sudo cp -r tensorflow/core /usr/local/tensorflow/include/tensorflow

sudo mkdir /usr/local/tensorflow/include/third_party
sudo cp -r third_party/eigen3 /usr/local/tensorflow/include/third_party/

sudo mkdir /usr/local/tensorflow/lib
sudo cp bazel-bin/tensorflow/libtensorflow_*.so /usr/local/tensorflow/lib