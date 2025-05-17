#include "amul.h"
#include "float3.h"
#include <stdint.h>

// Landau-Lifshitz torque.
extern "C" __global__ void
lltorque4D(float* __restrict__ n, float* __restrict__ l, 
           float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__  hx, float* __restrict__  hy, float* __restrict__  hz,
          float* __restrict__  alpha_, float alpha_mul, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 m = make_float3(mx[i], my[i], mz[i]);
        float3 H = make_float3(hx[i], hy[i], hz[i]);
        float alpha = amul(alpha_, alpha_mul, i);

        float3 mxH = cross(m, H);
        float gilb = -1.0f / (1.0f + alpha * alpha);

        float mB = mx[i]*hx[i] + my[i]*hy[i] + mz[i]*hz[i] + n[i]*l[i];

        // float3 torque = gilb * (mxH + alpha * cross(m, mxH));

        tx[i] = gilb*(mxH.x + alpha*(mB*m.x - H.x));
        ty[i] = gilb*(mxH.y + alpha*(mB*m.y - H.y));
        tz[i] = gilb*(mxH.z + alpha*(mB*m.z - H.z));
        l[i]  = gilb*alpha*(mB*n[i] - l[i]);
    }
}

