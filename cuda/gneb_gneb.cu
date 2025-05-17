#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

#define MAX(i, j) ( (i>j)? (i)  :  (j) )
#define MIN(i, j) ( (i<j)? (i)  :  (j) )
#define ABS(a) ( (a>0.0)? (a)  :  (-1.0*a) )

// See gneb.go for more details.




extern "C" __global__ void
gneb(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    float* __restrict__ Tx, float* __restrict__ Ty, float* __restrict__ Tz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            int Nx, int Ny, int Nz,int noi,int image, float Tp,
            float Lp,float Ln,float k,int CIGNEB, int Pos){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    if(iz/(Nz/noi) != image){
        return;
    }

 
    int I = idx(ix, iy, iz);

    float3 T  = make_float3(Tx[I]/sqrtf(Tp),Ty[I]/sqrtf(Tp),Tz[I]/sqrtf(Tp));
    float3 m0 = make_float3(mx[I], my[I], mz[I]);
    float3 B  = make_float3(Bx[I], By[I], Bz[I]);
    
    float pref;
    if(CIGNEB == 1 && iz/(Nz/noi) == Pos){
        pref = -2.*dot(B,T);
    }else{
        pref = k*(Ln-Lp) + dot(B,T);
    }
    
    B.x = B.x + pref*T.x;
    B.y = B.y + pref*T.y;
    B.z = B.z + pref*T.z;
    
    
    
    // The perpendicular component of the energy gradient
    // B = B - dot(B,T)*T;
    // T = k*(sqrtf(Lp)-sqrtf(Ln))*T + B;
    // T = k*(Lp-Ln)*T + B;

    Bx[I] = B.x - dot(m0,B)*m0.x;
    By[I] = B.y - dot(m0,B)*m0.y;
    Bz[I] = B.z - dot(m0,B)*m0.z;

    
}


