#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// See exchange.go for more details.
extern "C" __global__ void
addexchange_frustrated(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
            float* __restrict__ Sx, float* __restrict__ Sy, float* __restrict__ Sz,
            float* __restrict__ Ms_, float Ms_mul,
            float* __restrict__ aLUT2d, uint8_t* __restrict__ regions,
            float wx, float wy, float wz, int Nx, int Ny, int Nz, uint8_t PBC, 
            float J1, float J2, float J3, float J4) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx || j >= Ny || k >= Nz) {
        return;
    }

    int I = idx(i, j, k), np, nm;
    int nnm1, nnm2, nnp1, nnp2;
    float sx = Sx[I], sy = Sy[I], sz = Sz[I];
    float bx = 0.0, by = 0.0, bz = 0.0;
    


    //Shell 1 (6 neighbours)
    if(Nx>1){
        nm = indexPBC(i-1, j, k, Nx, Ny, Nz);
        np = indexPBC(i+1, j, k, Nx, Ny, Nz);

        bx += Sx[nm] + Sx[np] - 2.0*sx;
        by += Sy[nm] + Sy[np] - 2.0*sy;
        bz += Sz[nm] + Sz[np] - 2.0*sz;
    }
    
    if(Ny>1){
        nm = indexPBC(i, j-1, k, Nx, Ny, Nz);
        np = indexPBC(i, j+1, k, Nx, Ny, Nz);

        bx += Sx[nm] + Sx[np] - 2.0*sx;
        by += Sy[nm] + Sy[np] - 2.0*sy;
        bz += Sz[nm] + Sz[np] - 2.0*sz;
    }
    
    if(Nz>1){
        nm = indexPBC(i, j, k-1, Nx, Ny, Nz);
        np = indexPBC(i, j, k+1, Nx, Ny, Nz);

        bx += Sx[nm] + Sx[np] - 2.0*sx;
        by += Sy[nm] + Sy[np] - 2.0*sy;
        bz += Sz[nm] + Sz[np] - 2.0*sz;
    }
    

    Bx[I] += bx*J1; By[I] += by*J1; Bz[I] += bz*J1;

    //Shell 2 (12 neighbours)
    bx = 0.0; by = 0.0; bz = 0.0;
    if(Nx>1 && Ny>1){
        nnm1 = indexPBC(i-1, j-1, k, Nx, Ny, Nz);
        nnp1 = indexPBC(i-1, j+1, k, Nx, Ny, Nz);
        nnm2 = indexPBC(i+1, j-1, k, Nx, Ny, Nz);
        nnp2 = indexPBC(i+1, j+1, k, Nx, Ny, Nz);
        
        bx += Sx[nnm1] + Sx[nnp1] + Sx[nnm2] + Sx[nnp2] - 4.0*sx;
        by += Sy[nnm1] + Sy[nnp1] + Sy[nnm2] + Sy[nnp2] - 4.0*sy;
        bz += Sz[nnm1] + Sz[nnp1] + Sz[nnm2] + Sz[nnp2] - 4.0*sz;            
    }
    if(Nx>1 && Nz>1){
        nnm1 = indexPBC(i-1, j, k-1, Nx, Ny, Nz);
        nnp1 = indexPBC(i-1, j, k+1, Nx, Ny, Nz);
        nnm2 = indexPBC(i+1, j, k-1, Nx, Ny, Nz);
        nnp2 = indexPBC(i+1, j, k+1, Nx, Ny, Nz);
        
        bx += Sx[nnm1] + Sx[nnp1] + Sx[nnm2] + Sx[nnp2] - 4.0*sx;
        by += Sy[nnm1] + Sy[nnp1] + Sy[nnm2] + Sy[nnp2] - 4.0*sy;
        bz += Sz[nnm1] + Sz[nnp1] + Sz[nnm2] + Sz[nnp2] - 4.0*sz;            
    }
    if(Ny>1 && Nz>1){
        nnm1 = indexPBC(i, j-1, k-1, Nx, Ny, Nz);
        nnp1 = indexPBC(i, j-1, k+1, Nx, Ny, Nz);
        nnm2 = indexPBC(i, j+1, k-1, Nx, Ny, Nz);
        nnp2 = indexPBC(i, j+1, k+1, Nx, Ny, Nz);
        
        bx += Sx[nnm1] + Sx[nnp1] + Sx[nnm2] + Sx[nnp2] - 4.0*sx;
        by += Sy[nnm1] + Sy[nnp1] + Sy[nnm2] + Sy[nnp2] - 4.0*sy;
        bz += Sz[nnm1] + Sz[nnp1] + Sz[nnm2] + Sz[nnp2] - 4.0*sz;            
    }
    Bx[I] += bx*J2; By[I] += by*J2; Bz[I] += bz*J2;

    //Shell 3 (8 neighbours)
    bx = 0.0; by = 0.0; bz = 0.0;
    if(Nx>1 && Ny>1 && Ny>1){
        nnm1 = indexPBC(i-1, j-1, k-1, Nx, Ny, Nz);
        nnp1 = indexPBC(i-1, j+1, k-1, Nx, Ny, Nz);
        nnm2 = indexPBC(i+1, j-1, k-1, Nx, Ny, Nz);
        nnp2 = indexPBC(i+1, j+1, k-1, Nx, Ny, Nz);

        bx += Sx[nnm1] + Sx[nnp1] + Sx[nnm2] + Sx[nnp2] - 4.0*sx;
        by += Sy[nnm1] + Sy[nnp1] + Sy[nnm2] + Sy[nnp2] - 4.0*sy;
        bz += Sz[nnm1] + Sz[nnp1] + Sz[nnm2] + Sz[nnp2] - 4.0*sz;
        
        nnm1 = indexPBC(i-1, j-1, k+1, Nx, Ny, Nz);
        nnp1 = indexPBC(i-1, j+1, k+1, Nx, Ny, Nz);
        nnm2 = indexPBC(i+1, j-1, k+1, Nx, Ny, Nz);
        nnp2 = indexPBC(i+1, j+1, k+1, Nx, Ny, Nz);
        
        bx += Sx[nnm1] + Sx[nnp1] + Sx[nnm2] + Sx[nnp2] - 4.0*sx;
        by += Sy[nnm1] + Sy[nnp1] + Sy[nnm2] + Sy[nnp2] - 4.0*sy;
        bz += Sz[nnm1] + Sz[nnp1] + Sz[nnm2] + Sz[nnp2] - 4.0*sz;
    }
    Bx[I] += bx*J3; By[I] += by*J3; Bz[I] += bz*J3;

    //Shell 4 (6 neighbours)
    bx = 0.0; by = 0.0; bz = 0.0;
    if(Nx>2){
        nm = indexPBC(i-2, j, k, Nx, Ny, Nz);
        np = indexPBC(i+2, j, k, Nx, Ny, Nz);

        bx += Sx[nm] + Sx[np] - 2.0*sx;
        by += Sy[nm] + Sy[np] - 2.0*sy;
        bz += Sz[nm] + Sz[np] - 2.0*sz;
    }
    
    if(Ny>2){
        nm = indexPBC(i, j-2, k, Nx, Ny, Nz);
        np = indexPBC(i, j+2, k, Nx, Ny, Nz);

        bx += Sx[nm] + Sx[np] - 2.0*sx;
        by += Sy[nm] + Sy[np] - 2.0*sy;
        bz += Sz[nm] + Sz[np] - 2.0*sz;
    }
    
    if(Nz>2){
        nm = indexPBC(i, j, k-2, Nx, Ny, Nz);
        np = indexPBC(i, j, k+2, Nx, Ny, Nz);

        bx += Sx[nm] + Sx[np] - 2.0*sx;
        by += Sy[nm] + Sy[np] - 2.0*sy;
        bz += Sz[nm] + Sz[np] - 2.0*sz;
    }
    Bx[I] += bx*J4; By[I] += by*J4; Bz[I] += bz*J4;
}
