package cuda

import (
	"github.com/kuchkin/mumax3-gneb/cuda/cu"
	"github.com/kuchkin/mumax3-gneb/data"
	"github.com/kuchkin/mumax3-gneb/util"
)

// Stores the necessary state to perform FFT-accelerated convolution
// with magnetostatic kernel (or other kernel of same symmetry).
type DemagConvolution struct {
	inputSize          [3]int            // 3D size of the input/output data
	imSize             [3]int            // 3D size of the input/output data
	realKernSize       [3]int            // Size of kernel and logical FFT size.
	imrealKernSize     [3]int            // Size of kernel and logical FFT size.
	fftKernLogicSize   [3]int            // logic size FFTed kernel, real parts only, we store less
	imfftKernLogicSize [3]int            // logic size FFTed kernel, real parts only, we store less
	fftRBuf            [3]*data.Slice    // FFT input buf; 2D: Z shares storage with X.
	fftRBuf2           [3]*data.Slice    // FFT input buf; 2D: Z shares storage with X.
	fftCBuf            [3]*data.Slice    // FFT output buf; 2D: Z shares storage with X.
	fftCBuf2           [3]*data.Slice    // FFT output buf; 2D: Z shares storage with X.
	kern               [3][3]*data.Slice // FFT kernel on device
	imkern             [3][3]*data.Slice // FFT kernel on device
	fwPlan             fft3DR2CPlan      // Forward FFT (1 component)
	bwPlan             fft3DC2RPlan      // Backward FFT (1 component)
	imfwPlan           fft3DR2CPlan      // Forward FFT (1 component)
	imbwPlan           fft3DC2RPlan      // Backward FFT (1 component)
}

// Initializes a convolution to evaluate the demag field for the given mesh geometry.
// Sanity-checked if test == true (slow-ish for large meshes).
func NewDemag(noi int, gneb byte, inputSize, PBC [3]int, kernel [3][3]*data.Slice, imkernel [3][3]*data.Slice, test bool) *DemagConvolution {
	c := new(DemagConvolution)
	c.inputSize = inputSize
	c.imSize = inputSize
	if gneb == 1 || gneb == 2 {
		c.imSize[Z] = c.imSize[Z] / noi
	}
	c.realKernSize = kernel[X][X].Size()
	c.imrealKernSize = imkernel[X][X].Size()
	c.imfftKernLogicSize = imkernel[X][X].Size()

	c.init(kernel, imkernel)
	if test {
		testConvolution(c, PBC, kernel)
	}
	return c
}

// Calculate the demag field of m * vol * Bsat, store result in B.
// 	m:    magnetization normalized to unit length
// 	vol:  unitless mask used to scale m's length, may be nil
// 	Bsat: saturation magnetization in Tesla
// 	B:    resulting demag field, in Tesla
func (c *DemagConvolution) Exec(noi int, gneb byte, B, m2, m, vol *data.Slice, Msat MSlice) {

	util.Argument(B.Size() == c.inputSize && m.Size() == c.inputSize)
	// gneb := M.Mesh().GNEB_code()
	if gneb == 2 {
		c.exec3Dgneb(noi, B, m, m2, vol, Msat)
	} else if gneb == 1 {
		c.exec2Dgneb(noi, B, m, m2, vol, Msat)
	} else {
		if c.is2D() {
			c.exec2D(B, m, vol, Msat)
		} else {
			c.exec3D(B, m, vol, Msat)
		}
	}
}

func (c *DemagConvolution) exec3D(outp, inp, vol *data.Slice, Msat MSlice) {

	for i := 0; i < 3; i++ { // FW FFT
		c.fwFFT(i, inp, vol, Msat)

	}

	// kern mul
	kernMulRSymm3D_async(c.fftCBuf,
		c.kern[X][X], c.kern[Y][Y], c.kern[Z][Z],
		c.kern[Y][Z], c.kern[X][Z], c.kern[X][Y],
		c.fftKernLogicSize[X], c.fftKernLogicSize[Y], c.fftKernLogicSize[Z])

	for i := 0; i < 3; i++ { // BW FFT
		c.bwFFT(i, outp)
	}
}

func (c *DemagConvolution) exec3Dgneb(noi int, outp, inp, m2, vol *data.Slice, Msat MSlice) {

	for k := 0; k < noi; k++ {
		copyImage_async(inp, m2, noi, k)
		// copyImage_async(vol, vol2, noi,k)
		for i := 0; i < 3; i++ {
			c.fwFFT2(i, m2, vol, Msat)
		}

		//kern mul
		kernMulRSymm3Dimage_async(c.fftCBuf2,
			c.imkern[X][X], c.imkern[Y][Y], c.imkern[Z][Z],
			c.imkern[Y][Z], c.imkern[X][Z], c.imkern[X][Y],
			c.imfftKernLogicSize[X], c.imfftKernLogicSize[Y], c.imfftKernLogicSize[Z])

		// kernMulRSymm3D_async(c.fftCBuf2,
		// c.imkern[X][X], c.imkern[Y][Y], c.imkern[Z][Z],
		// c.imkern[Y][Z], c.imkern[X][Z], c.imkern[X][Y],
		// c.imfftKernLogicSize[X], c.imfftKernLogicSize[Y], c.imfftKernLogicSize[Z])

		for i := 0; i < 3; i++ { // BW FFT
			c.bwFFT2(i, m2)
		}
		copyBeff_async(m2, outp, noi, k)

		// for i := 0; i < 3; i++ {
		// 	c.fftRBuf2[i].Free()
		// 	c.fftCBuf2[i].Free()
		// 	c.fftRBuf2[i] = nil
		// 	c.fftCBuf2[i] = nil
		// }

		// c.fftCBuf2[X] = NewSlice(1, fftR2COutputSizeFloats(c.imrealKernSize))
		// c.fftCBuf2[Y] = NewSlice(1, fftR2COutputSizeFloats(c.imrealKernSize))
		// c.fftCBuf2[Z] = NewSlice(1, fftR2COutputSizeFloats(c.imrealKernSize))
		// c.fftRBuf2[X] = NewSlice(1, c.imrealKernSize)
		// c.fftRBuf2[Y] = NewSlice(1, c.imrealKernSize)
		// c.fftRBuf2[Z] = NewSlice(1, c.imrealKernSize)

		// copyFFT_async(c.fftRBuf2,c.fftRBuf,noi,k,c.fftKernLogicSize[X], c.fftKernLogicSize[Y],c.fftKernLogicSize[Z])
	}
	// print("tuta\n")

	// kern mul
	// kernMulRSymm3D_async(c.fftCBuf,
	// 	c.kern[X][X], c.kern[Y][Y], c.kern[Z][Z],
	// 	c.kern[Y][Z], c.kern[X][Z], c.kern[X][Y],
	// 	c.fftKernLogicSize[X], c.fftKernLogicSize[Y], c.fftKernLogicSize[Z])

	// for i := 0; i < 3; i++ { // BW FFT
	// 	c.bwFFT(i, outp)
	// }
}

func (c *DemagConvolution) exec2D(outp, inp, vol *data.Slice, Msat MSlice) {
	// Convolution is separated into
	// a 1D convolution for z and a 2D convolution for xy.
	// So only 2 FFT buffers are needed at the same time.
	Nx, Ny := c.fftKernLogicSize[X], c.fftKernLogicSize[Y]

	// Z
	c.fwFFT(Z, inp, vol, Msat)
	kernMulRSymm2Dz_async(c.fftCBuf[Z], c.kern[Z][Z], Nx, Ny)
	c.bwFFT(Z, outp)

	// XY
	c.fwFFT(X, inp, vol, Msat)
	c.fwFFT(Y, inp, vol, Msat)
	kernMulRSymm2Dxy_async(c.fftCBuf[X], c.fftCBuf[Y],
		c.kern[X][X], c.kern[Y][Y], c.kern[X][Y], Nx, Ny)
	c.bwFFT(X, outp)
	c.bwFFT(Y, outp)
}

func (c *DemagConvolution) exec2Dgneb(noi int, outp, inp, m2, vol *data.Slice, Msat MSlice) {
	// Convolution is separated into
	// a 1D convolution for z and a 2D convolution for xy.
	// So only 2 FFT buffers are needed at the same time.
	Nx, Ny := c.imfftKernLogicSize[X], c.imfftKernLogicSize[Y]

	// Z
	// c.fwFFT(Z, inp, vol, Msat)
	// kernMulRSymm2Dz_async(c.fftCBuf[Z], c.kern[Z][Z], Nx, Ny)
	// c.bwFFT(Z, outp)

	// // XY
	// c.fwFFT(X, inp, vol, Msat)
	// c.fwFFT(Y, inp, vol, Msat)
	// kernMulRSymm2Dxy_async(c.fftCBuf[X], c.fftCBuf[Y],
	// 	c.kern[X][X], c.kern[Y][Y], c.kern[X][Y], Nx, Ny)
	// c.bwFFT(X, outp)
	// c.bwFFT(Y, outp)

	for k := 0; k < noi; k++ {
		copyImage_async(inp, m2, noi, k)
		// copyImage_async(vol, vol2, noi,k)
		//Z
		c.fwFFT2(Z, m2, vol, Msat)
		kernMulRSymm2Dz_async(c.fftCBuf2[Z], c.imkern[Z][Z], Nx, Ny)
		c.bwFFT2(Z, m2)

		// XY
		c.fwFFT2(X, m2, vol, Msat)
		c.fwFFT2(Y, m2, vol, Msat)
		kernMulRSymm2Dxy_async(c.fftCBuf2[X], c.fftCBuf2[Y],
			c.imkern[X][X], c.imkern[Y][Y], c.imkern[X][Y], Nx, Ny)
		c.bwFFT2(X, m2)
		c.bwFFT2(Y, m2)

		copyBeff_async(m2, outp, noi, k)

		// for i := 0; i < 3; i++ {
		// 	c.fftRBuf2[i].Free()
		// 	c.fftCBuf2[i].Free()
		// 	c.fftRBuf2[i] = nil
		// 	c.fftCBuf2[i] = nil
		// }

		// c.fftCBuf2[X] = NewSlice(1, fftR2COutputSizeFloats(c.imrealKernSize))
		// c.fftCBuf2[Y] = NewSlice(1, fftR2COutputSizeFloats(c.imrealKernSize))
		// c.fftCBuf2[Z] = NewSlice(1, fftR2COutputSizeFloats(c.imrealKernSize))
		// c.fftRBuf2[X] = NewSlice(1, c.imrealKernSize)
		// c.fftRBuf2[Y] = NewSlice(1, c.imrealKernSize)
		// c.fftRBuf2[Z] = NewSlice(1, c.imrealKernSize)

		// copyFFT_async(c.fftRBuf2,c.fftRBuf,noi,k,c.fftKernLogicSize[X], c.fftKernLogicSize[Y],c.fftKernLogicSize[Z])
	}

}

func (c *DemagConvolution) is2D() bool {
	return c.inputSize[Z] == 1
}

// zero 1-component slice
func zero1_async(dst *data.Slice) {
	cu.MemsetD32Async(cu.DevicePtr(uintptr(dst.DevPtr(0))), 0, int64(dst.Len()), stream0)
}

// forward FFT component i
func (c *DemagConvolution) fwFFT(i int, inp, vol *data.Slice, Msat MSlice) {
	zero1_async(c.fftRBuf[i])
	in := inp.Comp(i)
	copyPadMul(c.fftRBuf[i], in, vol, c.realKernSize, c.inputSize, Msat)
	c.fwPlan.ExecAsync(c.fftRBuf[i], c.fftCBuf[i])
}

// forward FFT component i
func (c *DemagConvolution) fwFFT2(i int, inp, vol *data.Slice, Msat MSlice) {
	zero1_async(c.fftRBuf2[i])
	// zero1_async(c.fftCBuf2[i])
	in := inp.Comp(i)
	copyPadMul(c.fftRBuf2[i], in, vol, c.imrealKernSize, c.imSize, Msat)
	c.imfwPlan.ExecAsync(c.fftRBuf2[i], c.fftCBuf2[i])
}

// backward FFT component i
func (c *DemagConvolution) bwFFT(i int, outp *data.Slice) {
	c.bwPlan.ExecAsync(c.fftCBuf[i], c.fftRBuf[i])
	out := outp.Comp(i)
	copyUnPad(out, c.fftRBuf[i], c.inputSize, c.realKernSize)
}
func (c *DemagConvolution) bwFFT2(i int, outp *data.Slice) {
	c.imbwPlan.ExecAsync(c.fftCBuf2[i], c.fftRBuf2[i])
	out := outp.Comp(i)
	copyUnPad(out, c.fftRBuf2[i], c.imSize, c.imrealKernSize)
}

func (c *DemagConvolution) init(realKern [3][3]*data.Slice, imrealKern [3][3]*data.Slice) {
	// init device buffers
	// 2D re-uses fftBuf[X] as fftBuf[Z], 3D needs all 3 fftBufs.
	nc := fftR2COutputSizeFloats(c.realKernSize)
	c.fftCBuf[X] = NewSlice(1, nc)
	c.fftCBuf[Y] = NewSlice(1, nc)
	c.fftCBuf2[X] = NewSlice(1, fftR2COutputSizeFloats(c.imrealKernSize))
	c.fftCBuf2[Y] = NewSlice(1, fftR2COutputSizeFloats(c.imrealKernSize))
	if c.is2D() {
		c.fftCBuf[Z] = c.fftCBuf[X]
		c.fftCBuf2[Z] = c.fftCBuf2[X]
	} else {
		c.fftCBuf[Z] = NewSlice(1, nc)
		c.fftCBuf2[Z] = NewSlice(1, fftR2COutputSizeFloats(c.imrealKernSize))
	}

	c.fftRBuf[X] = NewSlice(1, c.realKernSize)
	c.fftRBuf[Y] = NewSlice(1, c.realKernSize)
	c.fftRBuf2[X] = NewSlice(1, c.imrealKernSize)
	c.fftRBuf2[Y] = NewSlice(1, c.imrealKernSize)
	if c.is2D() {
		c.fftRBuf[Z] = c.fftRBuf[X]
		c.fftRBuf2[Z] = c.fftRBuf2[X]
	} else {
		c.fftRBuf[Z] = NewSlice(1, c.realKernSize)
		c.fftRBuf2[Z] = NewSlice(1, c.imrealKernSize)
	}

	// init FFT plans
	c.fwPlan = newFFT3DR2C(c.realKernSize[X], c.realKernSize[Y], c.realKernSize[Z])
	c.bwPlan = newFFT3DC2R(c.realKernSize[X], c.realKernSize[Y], c.realKernSize[Z])

	c.imfwPlan = newFFT3DR2C(c.imrealKernSize[X], c.imrealKernSize[Y], c.imrealKernSize[Z])
	c.imbwPlan = newFFT3DC2R(c.imrealKernSize[X], c.imrealKernSize[Y], c.imrealKernSize[Z])

	// init FFT kernel

	// logic size of FFT(kernel): store real parts only
	c.fftKernLogicSize = fftR2COutputSizeFloats(c.realKernSize)
	util.Assert(c.fftKernLogicSize[X]%2 == 0)
	c.fftKernLogicSize[X] /= 2

	c.imfftKernLogicSize = fftR2COutputSizeFloats(c.imrealKernSize)
	util.Assert(c.imfftKernLogicSize[X]%2 == 0)
	c.imfftKernLogicSize[X] /= 2

	// physical size of FFT(kernel): store only non-redundant part exploiting Y, Z mirror symmetry
	// X mirror symmetry already exploited: FFT(kernel) is purely real.
	physKSize := [3]int{c.fftKernLogicSize[X], c.fftKernLogicSize[Y]/2 + 1, c.fftKernLogicSize[Z]/2 + 1}

	imphysKSize := [3]int{c.imfftKernLogicSize[X], c.imfftKernLogicSize[Y]/2 + 1, c.imfftKernLogicSize[Z]/2 + 1}

	output := c.fftCBuf[0]
	input := c.fftRBuf[0]
	fftKern := data.NewSlice(1, physKSize)
	kfull := data.NewSlice(1, output.Size()) // not yet exploiting symmetry
	kfulls := kfull.Scalars()
	kCSize := physKSize
	kCSize[X] *= 2                     // size of kernel after removing Y,Z redundant parts, but still complex
	kCmplx := data.NewSlice(1, kCSize) // not yet exploiting X symmetry
	kc := kCmplx.Scalars()

	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ { // upper triangular part
			if realKern[i][j] != nil { // ignore 0's
				// FW FFT
				data.Copy(input, realKern[i][j])
				c.fwPlan.ExecAsync(input, output)
				data.Copy(kfull, output)

				// extract non-redundant part (Y,Z symmetry)
				for iz := 0; iz < kCSize[Z]; iz++ {
					for iy := 0; iy < kCSize[Y]; iy++ {
						for ix := 0; ix < kCSize[X]; ix++ {
							kc[iz][iy][ix] = kfulls[iz][iy][ix]
							// if(iz==5 && ix == 5&& iy == 5){ print(kc[iz][iy][ix], "\n")}
						}
					}
				}

				// extract real parts (X symmetry)
				scaleRealParts(fftKern, kCmplx, 1/float32(c.fwPlan.InputLen()))
				c.kern[i][j] = GPUCopy(fftKern)
			}
		}
	}

	////gneb

	imoutput := c.fftCBuf2[0]
	iminput := c.fftRBuf2[0]
	imfftKern := data.NewSlice(1, imphysKSize)
	imkfull := data.NewSlice(1, imoutput.Size()) // not yet exploiting symmetry
	imkfulls := imkfull.Scalars()
	imkCSize := imphysKSize
	imkCSize[X] *= 2                       // size of kernel after removing Y,Z redundant parts, but still complex
	imkCmplx := data.NewSlice(1, imkCSize) // not yet exploiting X symmetry
	imkc := imkCmplx.Scalars()

	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ { // upper triangular part
			// data.Copy(imrealKern[i][j], realKern[i][j])
			if realKern[i][j] != nil { // ignore 0's
				// FW FFT
				data.Copy(iminput, realKern[i][j])
				c.imfwPlan.ExecAsync(iminput, imoutput)
				data.Copy(imkfull, imoutput)

				// extract non-redundant part (Y,Z symmetry)
				for iz := 0; iz < imkCSize[Z]; iz++ {
					for iy := 0; iy < imkCSize[Y]; iy++ {
						for ix := 0; ix < imkCSize[X]; ix++ {
							imkc[iz][iy][ix] = imkfulls[iz][iy][ix]
							// if(iz==5 && ix == 5 && iy == 5){ print(imkc[iz][iy][ix], "\n")}
						}
					}
				}

				// extract real parts (X symmetry)
				scaleRealParts(imfftKern, imkCmplx, 1/float32(c.imfwPlan.InputLen()))
				c.imkern[i][j] = GPUCopy(imfftKern)
			}
		}
	}
	// print("***********************OPA******************************\n")
}

func (c *DemagConvolution) Free() {
	if c == nil {
		return
	}
	c.inputSize = [3]int{}
	c.realKernSize = [3]int{}
	c.imSize = [3]int{}
	c.imrealKernSize = [3]int{}
	for i := 0; i < 3; i++ {
		c.fftCBuf[i].Free()
		c.fftRBuf[i].Free()
		c.fftRBuf2[i].Free()
		c.fftCBuf2[i].Free()
		c.fftCBuf[i] = nil
		c.fftRBuf[i] = nil
		c.fftRBuf2[i] = nil
		c.fftCBuf2[i] = nil

		for j := 0; j < 3; j++ {
			c.kern[i][j].Free()
			c.kern[i][j] = nil
			c.imkern[i][j].Free()
			c.imkern[i][j] = nil
		}
		c.fwPlan.Free()
		c.bwPlan.Free()
		c.imfwPlan.Free()
		c.imbwPlan.Free()

		cudaCtx.SetCurrent()
	}
}
