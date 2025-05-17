#include "reduce.h"
#include "atomicf.h"
#include "float3.h"

#define proj(mz) ((mz>0.0)? (1.0) : (-1.0))
#define load_vecnorm2(i) (fabs(Bx[i]*(y[i]*y[i] + proj(z[i])*z[i]*(1.0 + proj(z[i])*z[i])) - By[i]*x[i]*y[i] - Bz[i]*x[i]*(proj(z[i]) + z[i]))+fabs(-Bx[i]*x[i]*y[i] + By[i]*(x[i]*x[i] + proj(z[i])*z[i]*(1.0 + proj(z[i])*z[i])) - Bz[i]*y[i]*(proj(z[i]) + z[i])))

extern "C" __global__ void
reducemaxvecnorm2gg(float* __restrict__ x, float* __restrict__ y, float* __restrict__ z, 
					float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
 					float* __restrict__ dst, float initVal, int n) {
    reduce(load_vecnorm2, fmax, atomicFmaxabs)
}


/*

// #define load_vecnorm2(i) \
// 	(Bx[i]*x[i]) + (By[i]*y[i]) +  (Bz[i]*z[i])
*/

///torque
///(sqrtf((By[i]*z[i]-Bz[i]*y[i])*(By[i]*z[i]-Bz[i]*y[i]) + (Bx[i]*z[i]-Bz[i]*x[i])*(Bx[i]*z[i]-Bz[i]*x[i]) +  (By[i]*x[i]-Bx[i]*y[i])*(By[i]*x[i]-Bx[i]*y[i])))
