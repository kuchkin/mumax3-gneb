package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/kuchkin/mumax3-gneb/cuda/cu"
	"github.com/kuchkin/mumax3-gneb/timer"
	"sync"
	"unsafe"
)

// CUDA handle for shiftmagz kernel
var shiftmagz_code cu.Function

// Stores the arguments for shiftmagz kernel invocation
type shiftmagz_args_t struct {
	arg_dstX   unsafe.Pointer
	arg_dstY   unsafe.Pointer
	arg_dstZ   unsafe.Pointer
	arg_srcX   unsafe.Pointer
	arg_srcY   unsafe.Pointer
	arg_srcZ   unsafe.Pointer
	arg_Nx     int
	arg_Ny     int
	arg_Nz     int
	arg_shz    int
	arg_clampL float32
	arg_clampR float32
	argptr     [12]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for shiftmagz kernel invocation
var shiftmagz_args shiftmagz_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	shiftmagz_args.argptr[0] = unsafe.Pointer(&shiftmagz_args.arg_dstX)
	shiftmagz_args.argptr[1] = unsafe.Pointer(&shiftmagz_args.arg_dstY)
	shiftmagz_args.argptr[2] = unsafe.Pointer(&shiftmagz_args.arg_dstZ)
	shiftmagz_args.argptr[3] = unsafe.Pointer(&shiftmagz_args.arg_srcX)
	shiftmagz_args.argptr[4] = unsafe.Pointer(&shiftmagz_args.arg_srcY)
	shiftmagz_args.argptr[5] = unsafe.Pointer(&shiftmagz_args.arg_srcZ)
	shiftmagz_args.argptr[6] = unsafe.Pointer(&shiftmagz_args.arg_Nx)
	shiftmagz_args.argptr[7] = unsafe.Pointer(&shiftmagz_args.arg_Ny)
	shiftmagz_args.argptr[8] = unsafe.Pointer(&shiftmagz_args.arg_Nz)
	shiftmagz_args.argptr[9] = unsafe.Pointer(&shiftmagz_args.arg_shz)
	shiftmagz_args.argptr[10] = unsafe.Pointer(&shiftmagz_args.arg_clampL)
	shiftmagz_args.argptr[11] = unsafe.Pointer(&shiftmagz_args.arg_clampR)
}

// Wrapper for shiftmagz CUDA kernel, asynchronous.
func k_shiftmagz_async(dstX unsafe.Pointer, dstY unsafe.Pointer, dstZ unsafe.Pointer, srcX unsafe.Pointer, srcY unsafe.Pointer, srcZ unsafe.Pointer, Nx int, Ny int, Nz int, shz int, clampL float32, clampR float32, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("shiftmagz")
	}

	shiftmagz_args.Lock()
	defer shiftmagz_args.Unlock()

	if shiftmagz_code == 0 {
		shiftmagz_code = fatbinLoad(shiftmagz_map, "shiftmagz")
	}

	shiftmagz_args.arg_dstX = dstX
	shiftmagz_args.arg_dstY = dstY
	shiftmagz_args.arg_dstZ = dstZ
	shiftmagz_args.arg_srcX = srcX
	shiftmagz_args.arg_srcY = srcY
	shiftmagz_args.arg_srcZ = srcZ
	shiftmagz_args.arg_Nx = Nx
	shiftmagz_args.arg_Ny = Ny
	shiftmagz_args.arg_Nz = Nz
	shiftmagz_args.arg_shz = shz
	shiftmagz_args.arg_clampL = clampL
	shiftmagz_args.arg_clampR = clampR

	args := shiftmagz_args.argptr[:]
	cu.LaunchKernel(shiftmagz_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("shiftmagz")
	}
}

// maps compute capability on PTX code for shiftmagz kernel.
var shiftmagz_map = map[int]string{0: "",
	50: shiftmagz_ptx_50}

// shiftmagz PTX code for various compute capabilities.
const (
	shiftmagz_ptx_50 = `
.version 7.5
.target sm_50
.address_size 64

	// .globl	shiftmagz

.visible .entry shiftmagz(
	.param .u64 shiftmagz_param_0,
	.param .u64 shiftmagz_param_1,
	.param .u64 shiftmagz_param_2,
	.param .u64 shiftmagz_param_3,
	.param .u64 shiftmagz_param_4,
	.param .u64 shiftmagz_param_5,
	.param .u32 shiftmagz_param_6,
	.param .u32 shiftmagz_param_7,
	.param .u32 shiftmagz_param_8,
	.param .u32 shiftmagz_param_9,
	.param .f32 shiftmagz_param_10,
	.param .f32 shiftmagz_param_11
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<12>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<21>;


	ld.param.u64 	%rd1, [shiftmagz_param_0];
	ld.param.u64 	%rd2, [shiftmagz_param_1];
	ld.param.u64 	%rd3, [shiftmagz_param_2];
	ld.param.u64 	%rd4, [shiftmagz_param_3];
	ld.param.u64 	%rd5, [shiftmagz_param_4];
	ld.param.u64 	%rd6, [shiftmagz_param_5];
	ld.param.u32 	%r5, [shiftmagz_param_6];
	ld.param.u32 	%r6, [shiftmagz_param_7];
	ld.param.u32 	%r7, [shiftmagz_param_8];
	ld.param.u32 	%r8, [shiftmagz_param_9];
	ld.param.f32 	%f9, [shiftmagz_param_10];
	ld.param.f32 	%f8, [shiftmagz_param_11];
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r10, %r9, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_5;

	sub.s32 	%r4, %r3, %r8;
	setp.lt.s32 	%p6, %r4, 0;
	mov.f32 	%f10, %f9;
	mov.f32 	%f11, %f9;
	@%p6 bra 	$L__BB0_4;

	setp.ge.s32 	%p7, %r4, %r7;
	mov.f32 	%f9, %f8;
	mov.f32 	%f10, %f8;
	mov.f32 	%f11, %f8;
	@%p7 bra 	$L__BB0_4;

	mad.lo.s32 	%r18, %r4, %r6, %r2;
	mad.lo.s32 	%r19, %r18, %r5, %r1;
	cvta.to.global.u64 	%rd7, %rd4;
	mul.wide.s32 	%rd8, %r19, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f11, [%rd9];
	cvta.to.global.u64 	%rd10, %rd5;
	add.s64 	%rd11, %rd10, %rd8;
	ld.global.nc.f32 	%f10, [%rd11];
	cvta.to.global.u64 	%rd12, %rd6;
	add.s64 	%rd13, %rd12, %rd8;
	ld.global.nc.f32 	%f9, [%rd13];

$L__BB0_4:
	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	cvta.to.global.u64 	%rd14, %rd1;
	mul.wide.s32 	%rd15, %r21, 4;
	add.s64 	%rd16, %rd14, %rd15;
	st.global.f32 	[%rd16], %f11;
	cvta.to.global.u64 	%rd17, %rd2;
	add.s64 	%rd18, %rd17, %rd15;
	st.global.f32 	[%rd18], %f10;
	cvta.to.global.u64 	%rd19, %rd3;
	add.s64 	%rd20, %rd19, %rd15;
	st.global.f32 	[%rd20], %f9;

$L__BB0_5:
	ret;

}

`
)
