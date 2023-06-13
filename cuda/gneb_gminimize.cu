#include <stdint.h>
#include "float3.h"



// Descent energy minimizer
extern "C" __global__ void
gminimize(float* __restrict__ mx,  float* __restrict__  my,  float* __restrict__ mz,
         float* __restrict__ Bx,  float* __restrict__  By,  float* __restrict__ Bz,
         float dt, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    bool pp = true;
    
    if (i < N) {

        // if((((i/(N/Nz))/(Nz/noi)) == 0 ) && MinimizeFirst == 0) pp = false;
        // if((((i/(N/Nz))/(Nz/noi)) == (noi-1) ) && MinimizeLast == 0) pp = false;
        // if(MinimizeEndPoints == 1){
        //     pp = false;
        //     if((((i/(N/Nz))/(Nz/noi)) == 0 ) && MinimizeFirst == 1) pp = true;
        //     if((((i/(N/Nz))/(Nz/noi)) == (noi-1) ) && MinimizeLast == 1) pp = true;
        // }
        // if(pp){
            float3 m = {mx[i], my[i], mz[i]};
            float3 B = {Bx[i], By[i], Bz[i]};

            float ss = 1.0;
            if(m.z<0) ss = -1.0;


            
            float g1 = m.x/(1.0 + ss*m.z);
            float g2 = m.y/(1.0 + ss*m.z);
            float d1 = (B.x*(m.y*m.y + ss*m.z*(1.0 + ss*m.z)) - B.y*m.x*m.y - B.z*m.x*(ss + m.z));
            float d2 = (-B.x*m.x*m.y + B.y*(m.x*m.x + ss*m.z*(1.0 + ss*m.z)) - B.z*m.y*(ss + m.z));
            g1 = g1 + dt*d1;
            g2 = g2 + dt*d2;
            d1 = 1./(1.+g1*g1+g2*g2);
            mx[i] = 2.*g1*d1;
            my[i] = 2.*g2*d1;
            mz[i] = ss*(1.-g1*g1-g2*g2)*d1;
        // }
        

        // m += dt*B;
        // m = normalized(m);
        // mx[i] = m.x;
        // my[i] = m.y;
        // mz[i] = m.z;


    }
}
