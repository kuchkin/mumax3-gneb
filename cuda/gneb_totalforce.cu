#include "reduceforce.h"
#include "sum.h"

#define load(i) src[i]

extern "C" __global__ void
totalforce(float* __restrict__ src, float*__restrict__  dst, float initVal, int n, int noi,int Nz) {
    reduceforce(load, sum, atomicAdd,noi)
}