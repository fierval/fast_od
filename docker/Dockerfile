FROM fierval/tensorrt:19.02-py3

# nvcuvid
ADD nvcuvid/* /usr/local/cuda/targets/x86_64-linux/include/

# opencv
RUN apt-get update
RUN apt-get install -y git libgtk2.0-dev curl pkg-config autoconf automake libtool libavcodec-dev \
libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev \
libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev unzip libcurl4-gnutls-dev zlib1g-dev

RUN apt-get install -y wget

RUN apt-get install -y vim

## CMAKE
ADD https://cmake.org/files/v3.13/cmake-3.13.0.tar.gz /
RUN tar xzvf /cmake-3.13.0.tar.gz -C / \
    && cd /cmake-3.13.0 \
    && ./bootstrap \
    && make -j15 \
    && make install

# Second: get and build OpenCV 3.3.1
#
ADD https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-cpp-3.6.1.tar.gz /
RUN tar xzvf /protobuf-cpp-3.6.1.tar.gz -C /
RUN cd /protobuf-3.6.1 \
    && ./configure \
    && make -j15 \
    && make install \
    && ldconfig

ADD https://github.com/opencv/opencv_contrib/archive/3.3.1.zip  / 
RUN unzip -o /3.3.1.zip

ADD https://github.com/opencv/opencv/archive/3.3.1.zip / 
RUN unzip -o /3.3.1.zip 
RUN cd /workspace/opencv-3.3.1 \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_TIFF=ON \
       -DBUILD_opencv_java=OFF \
       -DBUILD_SHARED_LIBS=OFF \
       -DWITH_CUDA=ON \
       -DBUILD_PERF_TESTS=OFF \
       -DBUILD_TESTS=OFF \
       -DBUILD_opencv_codacodec=ON \
       # -DENABLE_FAST_MATH=1 \
       # -DCUDA_FAST_MATH=1 \
       -DWITH_CUBLAS=1 \
       -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
       -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.3.1/modules/ \
       ##
       -DCUDA_ARCH_BIN='7.0' \
       -DCUDA_ARCH_PTX="" \
       ##
       ## AVX in dispatch because not all machines have it
       -DCPU_DISPATCH=AVX,AVX2 \
       -DENABLE_PRECOMPILED_HEADERS=OFF \
       -DWITH_OPENGL=OFF \
       -DWITH_OPENCL=OFF \
       -DWITH_QT=OFF \
       -DWITH_NVCUVID=ON \
       -DWITH_IPP=ON \
       -DWITH_TBB=ON \
       -DFORCE_VTK=ON \
       -DWITH_EIGEN=ON \
       -DWITH_V4L=ON \
       -DWITH_XINE=ON \
       -DWITH_GDAL=ON \
       -DWITH_1394=OFF \
       -DWITH_FFMPEG=OFF \
       -DBUILD_PROTOBUF=ON \
       -DBUILD_TESTS=OFF \
       -DBUILD_PERF_TESTS=OFF \
       -DBUILD_opencv_xfeatures2d=OFF \
       -DCMAKE_BUILD_TYPE=RELEASE \
       -DCMAKE_INSTALL_PREFIX=/usr/local \
    .. \
    && make -j15 \
    && make install \
    && rm /3.3.1.zip \
    && rm /cmake-3.13.0.tar.gz

RUN mkdir fast_od
RUN mkdir /home/boris

# tensorflow libraries
ADD tensorflow.tar /

