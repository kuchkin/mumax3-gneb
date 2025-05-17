#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

#define MAX(a, b) ( (a>b)? (a)  :  (b) )
#define MIN(a, b) ( (a<b)? (a)  :  (b) )
#define ABS(a) ( (a>0.0)? (a)  :  (-1.0*a) )

// See gneb.go for more details.




extern "C" __global__ void
tangent(float* __restrict__ Tx, float* __restrict__ Ty, float* __restrict__ Tz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            int Nx, int Ny, int Nz,int noi,int image, float Ep, float Ei, float En,
            float Lp,float Ln){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    if(iz/(Nz/noi) != image){
        return;
    }

 

    float3 Tp  = make_float3(0.0,0.0,0.0);
    float3 Tm  = make_float3(0.0,0.0,0.0);
    float3 T   = make_float3(0.0,0.0,0.0);

    int I = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[I], my[I], mz[I]);
    int i_  = idx(ix, iy, iz-Nz/noi);
    float3 m1 = make_float3(mx[i_], my[i_], mz[i_]);
    i_  = idx(ix, iy, iz+Nz/noi);
    float3 m2 = make_float3(mx[i_], my[i_], mz[i_]);

    Tp = (m2-m0);
    Tm = (m0-m1);
    // Tp.x /= (Ln);
    // Tp.y /= (Ln);
    // Tp.z /= (Ln);
    // Tm.x /= (Lp);
    // Tm.y /= (Lp);
    // Tm.z /= (Lp);
    if(En>Ei && Ei>Ep){
        T = Tp;
    }else if(En<Ei && Ei<Ep){
        T = Tm;
    }else{
        float dEmax = MAX(ABS(En-Ei),ABS(Ep-Ei))+1e-8;
        float dEmin = MIN(ABS(En-Ei),ABS(Ep-Ei));
        // float pref  = 1.0;
        // if(sqrt(dEmax*dEmax+dEmin*dEmin)>0) pref = 1./sqrt(dEmax*dEmax+dEmin*dEmin);
        if(En>Ep){
            // T = (dEmax*Tp + dEmin*Tm)*pref;
            T = Tp + (dEmin/dEmax)*Tm;
            // T = Tp + (dEmin)*Tm;
        }else{
            //T = (dEmin*Tp + dEmax*Tm)*pref;
            T = (dEmin/dEmax)*Tp + Tm;
            // T = (dEmin)*Tp + Tm;
        }
    }

    float m0T = dot(m0,T);

    Tx[I] = T.x - m0T*m0.x;
    Ty[I] = T.y - m0T*m0.y;
    Tz[I] = T.z - m0T*m0.z;

    
}


