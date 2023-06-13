// 3D micromagnetic kernel multiplication:
//
// |Mx|   |Kxx Kxy Kxz|   |Mx|
// |My| = |Kxy Kyy Kyz| * |My|
// |Mz|   |Kxz Kyz Kzz|   |Mz|
//
// ~kernel has mirror symmetry along Y and Z-axis,
// apart form first row,
// and is only stored (roughly) half:
//
// K11, K22, K02:
// xxxxx
// aaaaa
// bbbbb
// ....
// bbbbb
// aaaaa
//
// K12:
// xxxxx
// aaaaa
// bbbbb
// ...
// -bbbb
// -aaaa

extern "C" __global__ void
copyFFT(float* __restrict__  fft2x,  float* __restrict__  fft2y,  float* __restrict__  fft2z,
                float* __restrict__  fftx,  float* __restrict__  ffty,  float* __restrict__  fftz,
               int noi, int image, int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix>= Nx || iy>= Ny || iz>=Nz) {
        return;
    }
    if(iz/(Nz/noi) != image) return;

    // fetch (complex) FFT'ed magnetization
    int I = (iz*Ny + iy)*Nx + ix;
    int e = 1 * I;
    

    // m * K matrix multiplication, overwrite m with result.
    fftx[e  ] = fft2x[e  ] ;
    // fftx[e+1] = fft2x[e+1] ;
    ffty[e  ] = fft2y[e  ] ;
    // ffty[e+1] = fft2y[e+1] ;
    fftz[e  ] = fft2z[e  ] ;
    // fftz[e+1] = fft2z[e+1] ;
}

