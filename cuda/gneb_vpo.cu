#include <stdint.h>
#include "float3.h"



// Descent energy minimizer
extern "C" __global__ void
vpo(float* __restrict__ mx,  float* __restrict__  my,  float* __restrict__ mz,
         float* __restrict__ Bx,  float* __restrict__  By,  float* __restrict__ Bz,uint8_t* regions,
          float dt, float vf, int minend, int noi, int N, int Nz) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    // bool pp = true;
    
    if (i < N) {


        //search direction
        // float vf = 0.0;
        // if((((i/(N/Nz))/(Nz/noi)) == 0 )){
        //     vf = vf1; //first image
        // }else if((((i/(N/Nz))/(Nz/noi)) == (noi-1) )){
        //     vf = vf3; //last image
        // }else{
        //     vf = vf2; //intermadiate images
        // }

        Bx[i] = Bx[i]*vf; By[i] = By[i]*vf; Bz[i] = Bz[i]*vf;

        float3 m = {mx[i], my[i], mz[i]};
        float3 B = {Bx[i], By[i], Bz[i]};



        //simple
        // m += dt*B;
        // m = normalized(m);
        bool p;
        if(minend == 0 ){
            p = true;
            // p = ( (((i/(N/Nz))/(Nz/noi)) == 0 ) || (((i/(N/Nz))/(Nz/noi)) == (noi-1) ) )? false : true;
        }else{
            p = ( (((i/(N/Nz))/(Nz/noi)) == 0 ) || (((i/(N/Nz))/(Nz/noi)) == (noi-1) ) )? true : false;
        }

        //rotation rodrigues

        if(p){


        float theta = dt*len(B);
        float pref;
        if(theta<=0.005){//1e-5
            pref = 1.0 - theta*theta*(1.0-theta*theta/20.0)/6.0;
            
        }else{
            pref = sin(theta)/theta;
        }

        pref *= dt;

        
        m = m*cos(theta) + B*pref;
        if(regions[i] == 255){
            m.x = 0.0; m.y = 0.0; m.z = 1.0;
        }
        
        m = normalized(m);
        mx[i] = m.x;
        my[i] = m.y;
        mz[i] = m.z;
        }

        //rotation matrix
        // float theta = dt*len(B);

        // float q = cos(theta), w = 1-q;
        // float x = -dt*Bx[i]/theta, y = -dt*By[i]/theta, z = -dt*Bz[i]/theta;
        // float s1 = -y*z*w, s2 = x*z*w, s3 = -x*y*w;
        // float p1 = x*sin(theta), p2 = y*sin(theta), p3 = z*sin(theta);

        // float t1, t2, t3;
        // if(theta > 1.0e-20){
        //     t1 = (q+z*z*w) * mx[i] + (s1+p1)   * my[i] + (s2+p2)   * mz[i];
        //     t2 = (s1-p1)   * mx[i] + (q+y*y*w) * my[i] + (s3+p3)   * mz[i];
        //     t3 = (s2-p2)   * mx[i] + (s3-p3)   * my[i] + (q+x*x*w) * mz[i];
        //     mx[i] = t1;
        //     my[i] = t2;
        //     mz[i] = t3;
        // };

        //rotation
        // float theta = dt*len(B);
        // if(theta < 1e-20){
        //     m += dt*B;
        //     m = normalized(m);
        // }else{
        //     float3 vecnom = cross(m,B)*(1./len(B));
        //     m = m*cos(theta) + cross(vecnom,m)*sin(theta);
        // }

        // mx[i] = m.x;
        // my[i] = m.y;
        // mz[i] = m.z;
        


    }
}
