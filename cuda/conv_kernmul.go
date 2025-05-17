package cuda

// Kernel multiplication for purely real kernel, symmetric around Y axis (apart from first row).
// Launch configs range over all complex elements of fft input. This could be optimized: range only over kernel.

import (
	"github.com/kuchkin/mumax3-gneb/data"
	"github.com/kuchkin/mumax3-gneb/util"
)

// kernel multiplication for 3D demag convolution, exploiting full kernel symmetry.
func kernMulRSymm3D_async(fftM [3]*data.Slice, Kxx, Kyy, Kzz, Kyz, Kxz, Kxy *data.Slice, Nx, Ny, Nz int) {
	util.Argument(fftM[X].NComp() == 1 && Kxx.NComp() == 1)

	cfg := make3DConf([3]int{Nx, Ny, Nz})
	k_kernmulRSymm3D_async(fftM[X].DevPtr(0), fftM[Y].DevPtr(0), fftM[Z].DevPtr(0),
		Kxx.DevPtr(0), Kyy.DevPtr(0), Kzz.DevPtr(0), Kyz.DevPtr(0), Kxz.DevPtr(0), Kxy.DevPtr(0),
		Nx, Ny, Nz, cfg)
}

// kernel multiplication for 2D demag convolution on X and Y, exploiting full kernel symmetry.
func kernMulRSymm2Dxy_async(fftMx, fftMy, Kxx, Kyy, Kxy *data.Slice, Nx, Ny int) {
	util.Argument(fftMy.NComp() == 1 && Kxx.NComp() == 1)

	cfg := make3DConf([3]int{Nx, Ny, 1})
	k_kernmulRSymm2Dxy_async(fftMx.DevPtr(0), fftMy.DevPtr(0),
		Kxx.DevPtr(0), Kyy.DevPtr(0), Kxy.DevPtr(0),
		Nx, Ny, cfg)
}

// kernel multiplication for 2D demag convolution on Z, exploiting full kernel symmetry.
func kernMulRSymm2Dz_async(fftMz, Kzz *data.Slice, Nx, Ny int) {
	util.Argument(fftMz.NComp() == 1 && Kzz.NComp() == 1)

	cfg := make3DConf([3]int{Nx, Ny, 1})
	k_kernmulRSymm2Dz_async(fftMz.DevPtr(0), Kzz.DevPtr(0), Nx, Ny, cfg)
}

// kernel multiplication for general 1D convolution. Does not assume any symmetry.
// Used for MFM images.
func kernMulC_async(fftM, K *data.Slice, Nx, Ny int) {
	util.Argument(fftM.NComp() == 1 && K.NComp() == 1)
	cfg := make3DConf([3]int{Nx, Ny, 1})
	k_kernmulC_async(fftM.DevPtr(0), K.DevPtr(0), Nx, Ny, cfg)
}

// copy image
func copyImage_async(m, m2 *data.Slice, noi, image int) {
	// util.Argument(fftM[X].NComp() == 1 && Kxx.NComp() == 1)
	NN := m2.Size()
	cfg := make3DConf([3]int{NN[X], NN[Y], NN[Z]})
	k_copyImage_async(m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		noi, image, NN[X], NN[Y], NN[Z], cfg)
}

// kernel multiplication for 3D demag convolution, exploiting full kernel symmetry, image only.
func kernMulRSymm3Dimage_async(fftM [3]*data.Slice, Kxx, Kyy, Kzz, Kyz, Kxz, Kxy *data.Slice, Nx, Ny, Nz int) {
	util.Argument(fftM[X].NComp() == 1 && Kxx.NComp() == 1)

	cfg := make3DConf([3]int{Nx, Ny, Nz})
	k_kernmulRSymm3Dimage_async(fftM[X].DevPtr(0), fftM[Y].DevPtr(0), fftM[Z].DevPtr(0),
		Kxx.DevPtr(0), Kyy.DevPtr(0), Kzz.DevPtr(0), Kyz.DevPtr(0), Kxz.DevPtr(0), Kxy.DevPtr(0),
		Nx, Ny, Nz, cfg)
}

func copyFFT_async(fft2, fft [3]*data.Slice, noi, image, Nx, Ny, Nz int) {
	// util.Argument(fftM[X].NComp() == 1 && Kxx.NComp() == 1)

	cfg := make3DConf([3]int{Nx, Ny, Nz})
	k_copyFFT_async(fft2[X].DevPtr(0), fft2[Y].DevPtr(0), fft2[Z].DevPtr(0),
		fft[X].DevPtr(0), fft[Y].DevPtr(0), fft[Z].DevPtr(0),
		noi, image, Nx, Ny, Nz, cfg)
}

// copy image
func copyBeff_async(B2, B *data.Slice, noi, image int) {
	// util.Argument(fftM[X].NComp() == 1 && Kxx.NComp() == 1)
	NN := B2.Size()
	cfg := make3DConf([3]int{NN[X], NN[Y], NN[Z]})
	k_copyBeff_async(B2.DevPtr(X), B2.DevPtr(Y), B2.DevPtr(Z),
		B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		noi, image, NN[X], NN[Y], NN[Z], cfg)
}
