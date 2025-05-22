#include <stdint.h>
#include "float3.h"



// Descent energy minimizer
extern "C" __global__ void
vpo4D(float* __restrict__ mx,  float* __restrict__  my,  float* __restrict__ mz, float* __restrict__ mw,
         float* __restrict__ Bx,  float* __restrict__  By,  float* __restrict__ Bz, float* __restrict__  Bw,
         uint8_t* regions, float dt,float vf, int minend, int noi, int N, int Nz) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    // bool pp = true;
    
    if (i < N) {

        // if((((i/(N/Nz))/(Nz/noi)) == 0 ) || (((i/(N/Nz))/(Nz/noi)) == (noi-1) )){

        //     float nx = mx[i], ny = my[i], nz = mz[i], nv = mw[i];
        //     float tx = Bx[i], ty = By[i], tz = Bz[i], tv = Bw[i];

        //     float d1,d2,d3;
            
        //     float ss  = (nv > 0.0)? 1.0 : 1.0;
        //     float den = (nv > 0.0)? 1.0/(1.0 + nv) : 1.0/(1.0 - nv);
            
        //     float g1 = nx*den, g2 = ny*den, g3 = nz*den; 

        //     d1 = (-tx*(1.0+ss*nv - nx*nx) + ty*nx*ny + tz*nx*nz + tv*nx*(ss+nv));
        //     d2 = (tx*nx*ny - ty*(1.0+ss*nv - ny*ny) + tz*ny*nz + tv*ny*(ss+nv));
        //     d3 = (tx*nx*nz + ty*ny*nz - tz*(1.0+ss*nv - nz*nz) + tv*nz*(ss+nv));

        //     g1 -= dt*d1; g2 -= dt*d2; g3 -= dt*d3;
            
        //     float gg = g1*g1+g2*g2+g3*g3;
        //     float gi = 1./(1.+gg);
        //     nx = 2.*g1*gi; ny = 2.*g2*gi; nz = 2.*g3*gi; nv = ss*(1.-gg)*gi;
        //     mx[i] = nx;
        //     my[i] = ny;
        //     mz[i] = nz;
        //     mw[i] = nv;

        // }else if(minend == 0){
        //     Bx[i] = Bx[i]*vf; By[i] = By[i]*vf; Bz[i] = Bz[i]*vf; Bw[i] = Bw[i]*vf;

        //     float3 m = {mx[i], my[i], mz[i]};
        //     float3 B = {Bx[i], By[i], Bz[i]};


        //     float theta = dt*len4D(B, Bw[i]);
        //     float pref;
        //     if(theta<=0.005){//1e-5
        //         pref = 1.0 - theta*theta*(1.0-theta*theta/20.0)/6.0;
                
        //     }else{
        //         pref = sin(theta)/theta;
        //     }

        //     pref *= dt;
            
        //     m = m*cos(theta) + B*pref;
        //     float m4 = mw[i]*cos(theta) + Bw[i]*pref;

        //     if(regions[i] == 255){
        //         m.x = 0.0; m.y = 0.0; m.z = 1.0; m4 = 0.0;
        //     }
        //     float norm = 1.0/len4D(m, m4);
        //     // m = normalized(m);
        //         mx[i] = m.x*norm;
        //         my[i] = m.y*norm;
        //         mz[i] = m.z*norm;
        //         mw[i] =  m4*norm;
        //     }

        Bx[i] = Bx[i]*vf; By[i] = By[i]*vf; Bz[i] = Bz[i]*vf; Bw[i] = Bw[i]*vf;

            float3 m = {mx[i], my[i], mz[i]};
            float3 B = {Bx[i], By[i], Bz[i]};


            float theta = dt*len4D(B, Bw[i]);
            float pref;
            if(theta<=0.005){//1e-5
                pref = 1.0 - theta*theta*(1.0-theta*theta/20.0)/6.0;
                
            }else{
                pref = sin(theta)/theta;
            }

            pref *= dt;
            
            m = m*cos(theta) + B*pref;
            float m4 = mw[i]*cos(theta) + Bw[i]*pref;

            if(regions[i] == 255){
                m.x = 0.0; m.y = 0.0; m.z = 1.0; m4 = 0.0;
            }
            float norm = 1.0/len4D(m, m4);
            // m = normalized(m);
                mx[i] = m.x*norm;
                my[i] = m.y*norm;
                mz[i] = m.z*norm;
                mw[i] =  m4*norm;
        }

}
