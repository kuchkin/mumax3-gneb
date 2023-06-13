#include "reduceim2.h"
#include "sum.h"

#define load(i) src[i]
// #define loadim(i,image, noi,Nz,n) ((((i/(n/Nz))/(Nz/noi)) == image)? (src[i]) : (0.0))


extern "C" __global__ void
sumimage(float* __restrict__ src, float*__restrict__  dst, float initVal, int n, int image, int noi, int Nz) {
    reduceim2(load, sum, atomicAdd,image,noi,Nz)
}

