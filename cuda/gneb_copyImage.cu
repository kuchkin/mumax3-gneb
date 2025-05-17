


extern "C" __global__ void
copyImage(float* __restrict__  Mx,  float* __restrict__  My,  float* __restrict__  Mz,
        float* __restrict__  Mx2,  float* __restrict__  My2,  float* __restrict__  Mz2,
        int noi, int image, int Nx, int Ny, int Nz){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix>= Nx || iy>= Ny || iz>=Nz) {
        return;
    }
    int I = (iz*Ny + iy)*Nx + ix;
    float pref = 1.0;
    // if(iz%3!=0) pref = 0.0;
    iz = iz + image*Nz;
    int II = (iz*Ny + iy)*Nx + ix;
    Mx2[I] = pref*Mx[II];
    My2[I] = pref*My[II];
    Mz2[I] = pref*Mz[II];
    // vol2[I] = vol[II];
    
    // Mx2[I] = 0.0;
    // My2[I] = 0.0;
    // Mz2[I] = 0.0;
    // if(iz/(Nz/noi)==image){
    //     int I = (iz*Ny + iy)*Nx + ix;
    //     Mx2[((iz%(Nz/noi))*Ny + iy)*Nx + ix] = Mx[I];
    //     My2[((iz%(Nz/noi))*Ny + iy)*Nx + ix] = My[I];
    //     Mz2[((iz%(Nz/noi))*Ny + iy)*Nx + ix] = Mz[I];
    // // }else{
    // //     Mx2[I] = 0.0;
    // //     My2[I] = 0.0;
    // //     Mz2[I] = 0.0;
    // }

}

