#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// See exchange.go for more details.
extern "C" __global__ void
addexchange4D(float* __restrict__ n, float* __restrict__ Bx,
            float* __restrict__ Ms_, float Ms_mul, float kappa,
            float* __restrict__ aLUT2d, uint8_t* __restrict__ regions,
            float wx, float wy, float wz, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // central cell
    int I = idx(ix, iy, iz);
    float3 m0 = make_float3(n[I], 0.0, 0.0);
    // if (is0(m0)) {
    //     return;
    // }

    uint8_t r0 = regions[I];
    float3 B  = make_float3(0.0,0.0,0.0);

    int i_;    // neighbor index
    float3 m_; // neighbor mag
    float a__; // inter-cell exchange stiffness

    // left neighbor
    i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(n[i_], 0.0, 0.0);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wx * a__ *(m_ - m0);

    // right neighbor
    i_  = idx(hclampx(ix+1), iy, iz);
    m_  = make_float3(n[i_], 0.0, 0.0);
    m_  = ( is0(m_)? m0: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wx * a__ *(m_ - m0);

    // back neighbor
    i_  = idx(ix, lclampy(iy-1), iz);
    m_  = make_float3(n[i_], 0.0, 0.0);
    m_  = ( is0(m_)? m0: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ *(m_ - m0);

    // front neighbor
    i_  = idx(ix, hclampy(iy+1), iz);
    m_  = make_float3(n[i_], 0.0, 0.0);
    m_  = ( is0(m_)? m0: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ *(m_ - m0);

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
        i_  = idx(ix, iy, lclampz(iz-1));
        m_  = make_float3(n[i_], 0.0, 0.0);
        m_  = ( is0(m_)? m0: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ *(m_ - m0);

        // top neighbor
        i_  = idx(ix, iy, hclampz(iz+1));
        m_  = make_float3(n[i_], 0.0, 0.0);
        m_  = ( is0(m_)? m0: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ *(m_ - m0);
    }

    float invMs = inv_Msat(Ms_, Ms_mul, I);
    Bx[I] = B.x*invMs;
    Bx[I] -= 2.0*kappa*n[I]*invMs;
    return;
}

