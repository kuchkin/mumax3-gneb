#include <stdint.h>
#include "float3.h"



// generate u1 u2
extern "C" __global__ void
generate_w(float* __restrict__ w2x,  float* __restrict__  w2y,  int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    // bool pp = true;
    
    if (i < N) {

        w2x[i] = 1.0/sqrt(2.0*N);
        w2y[i] = 1.0/sqrt(2.0*N);


    }
}
