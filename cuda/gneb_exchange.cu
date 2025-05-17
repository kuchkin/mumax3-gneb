#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// See exchange.go for more details.
extern "C" __global__ void
gneb_addexchange(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            float* __restrict__ Ms_, float Ms_mul,
            float* __restrict__ aLUT2d, uint8_t* __restrict__ regions,
            float wx, float wy, float wz, int Nx, int Ny, int Nz,int noi, uint8_t PBC, uint8_t GNEB, float JZ) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // central cell
    int I = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[I], my[I], mz[I]);

    if (is0(m0)) {
        return;
    }

    uint8_t r0 = regions[I];
    float3 B  = make_float3(0.0,0.0,0.0);
    // float3 T  = make_float3(0.0,0.0,0.0);

    int i_;    // neighbor index
    float3 m_; // neighbor mag
    float a__; // inter-cell exchange stiffness

    float3 m1;
    float3 m2;

    // left neighbor
    i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wx * a__ *(m_ - m0);

    // right neighbor
    i_  = idx(hclampx(ix+1), iy, iz);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    m_  = ( is0(m_)? m0: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wx * a__ *(m_ - m0);

    // back neighbor
    i_  = idx(ix, lclampy(iy-1), iz);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    m_  = ( is0(m_)? m0: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ *(m_ - m0);

    // front neighbor
    i_  = idx(ix, hclampy(iy+1), iz);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    m_  = ( is0(m_)? m0: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ *(m_ - m0);

    // only take vertical derivative for 3D sim
    if (Nz != 1 && !GNEB2D && !GNEB3D) {
        // bottom neighbor
        i_  = idx(ix, iy, lclampz(iz-1));
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        m_  = ( is0(m_)? m0: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ *(m_ - m0);

        // top neighbor
        i_  = idx(ix, iy, hclampz(iz+1));
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        m_  = ( is0(m_)? m0: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ *(m_ - m0);
    }
    //&& Nz%noi == 0
    if (Nz != 1 && GNEB3D ) {
        // bottom neighbor
        // i_  = idx(ix, iy, max(iz%(Nz/noi)-1,0) + (iz/(Nz/noi))*Nz/noi);
        i_  = idx(ix, iy, lclamp(iz%(Nz/noi)-1,Nz/noi) + (iz/(Nz/noi))*Nz/noi);
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        m_  = ( is0(m_)? m0: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += JZ*wz * a__ *(m_ - m0);

        // top neighbor
        // i_  = idx(ix, iy, min(iz%(Nz/noi)+1,Nz/noi-1) + (iz/(Nz/noi))*Nz/noi);
        i_  = idx(ix, iy, hclamp(iz%(Nz/noi)+1,Nz/noi) + (iz/(Nz/noi))*Nz/noi);
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        m_  = ( is0(m_)? m0: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += JZ*wz * a__ *(m_ - m0);
    }

    // if(GNEB2D && iz !=0 && iz != (Nz-1)){
    //     i_  = idx(ix, iy, iz-1);
    //     m1 = make_float3(mx[i_], my[i_], mz[i_]);
    //     i_  = idx(ix, iy, iz+1);
    //     m2 = make_float3(mx[i_], my[i_], mz[i_]);
    //     T = normalized((m2-m1) - dot((m2-m1), m0)*m0);
    // }

    float invMs = inv_Msat(Ms_, Ms_mul, I);
    float par = 1.0;
    // if(GNEB2D || GNEB3D) par = 0.5;
    Bx[I] += B.x*invMs*par;
    By[I] += B.y*invMs*par;
    Bz[I] += B.z*invMs*par;
}

