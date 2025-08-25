#ifndef _STENCIL_H_
#define _STENCIL_H_

// 3D array indexing
#define index(ix,iy,iz,Nx,Ny,Nz) ( ( (iz)*(Ny) + (iy) ) * (Nx) + (ix) )
#define idx(ix,iy,iz) ( index((ix),(iy),(iz),(Nx),(Ny),(Nz)) )

#define indexPBC(i, j, k, Nx, Ny, Nz) ((((i) + (Nx)) % (Nx)) + (((j) + (Ny)) % (Ny)) * (Nx) + (((k) + (Nz)) % (Nz)) * (Nx) * (Ny))


// modulo used for PBC wrap around
#define MOD(n, M) ( (( (n) % (M) ) + (M) ) % (M)  )

// have PBC in x, y or z?
#define PBCx (PBC & 1)
#define PBCy (PBC & 2)
#define PBCz (PBC & 4)

#define GNEB2D (GNEB & 1)
#define GNEB3D (GNEB & 2)


// clamp or wrap index at boundary, depending on PBC
// hclamp*: clamps on upper side (index+1)
// lclamp*: clamps on lower side (index-1)
// *clampx: clamps along x
// ...
#define hclampx(ix) (PBCx? MOD(ix, Nx) : min((ix), Nx-1))
#define lclampx(ix) (PBCx? MOD(ix, Nx) : max((ix), 0))

#define hclampy(iy) (PBCy? MOD(iy, Ny) : min((iy), Ny-1))
#define lclampy(iy) (PBCy? MOD(iy, Ny) : max((iy), 0))

#define hclampz(iz) (PBCz? MOD(iz, Nz) : min((iz), Nz-1))
#define lclampz(iz) (PBCz? MOD(iz, Nz) : max((iz), 0))

#define hclamp(i,n) (PBCz? MOD(i, n) : min((i), n-1))
#define lclamp(i,n) (PBCz? MOD(i, n) : max((i), 0))

// #define hclamp(i,n) ((i%n == (i-1+n)%n) ? max((i-1),0) : i)
// #define lclamp(i,n) ((i%n == (i+1)%n ) ? min((i+1),n-1) : i)

#endif

