// kernel to convert from OpenCV channel representation to channel-first
// see: https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#how-the-image-matrix-is-stored-in-the-memory

const int BLOCK_SIZE = 1024;
#include <cuda_runtime.h>

__global__ void channelFirstKernel(unsigned char * source, float * dest, int channelSize, int channelsNum, int rowElems, int rowSize)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = idx / channelsNum;
    int channel = idx % channelsNum;

    // what would the row be if we didn't have any padding
    int row = idx / rowElems;
    int col = idx % rowElems;

    // actual element - skip padding
    int sourceIdx = row * rowSize + col;
    dest[channelSize * channel + offset] = ((float) source[sourceIdx]) * (2.0/255.0) - 1.0;
}

// we expect all memory to already reside on device so no need to allocate anything
void channelFirst(unsigned char * source, float * dest, int channelSize, int channelsNum, int rowElems, int rowSize)
{
    int nBlocks = (channelSize * channelsNum + BLOCK_SIZE - 1) / BLOCK_SIZE;

    channelFirstKernel<<<nBlocks, BLOCK_SIZE>>>(source, dest, channelSize, channelsNum, rowElems, rowSize);
    cudaDeviceSynchronize();
}

