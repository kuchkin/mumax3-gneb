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

// CUDA handle for projection kernel
var projection_code cu.Function

// Stores the arguments for projection kernel invocation
type projection_args_t struct {
	arg_kx unsafe.Pointer
	arg_ky unsafe.Pointer
	arg_kz unsafe.Pointer
	arg_mx unsafe.Pointer
	arg_my unsafe.Pointer
	arg_mz unsafe.Pointer
	arg_N  int
	argptr [7]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for projection kernel invocation
var projection_args projection_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	projection_args.argptr[0] = unsafe.Pointer(&projection_args.arg_kx)
	projection_args.argptr[1] = unsafe.Pointer(&projection_args.arg_ky)
	projection_args.argptr[2] = unsafe.Pointer(&projection_args.arg_kz)
	projection_args.argptr[3] = unsafe.Pointer(&projection_args.arg_mx)
	projection_args.argptr[4] = unsafe.Pointer(&projection_args.arg_my)
	projection_args.argptr[5] = unsafe.Pointer(&projection_args.arg_mz)
	projection_args.argptr[6] = unsafe.Pointer(&projection_args.arg_N)
}

// Wrapper for projection CUDA kernel, asynchronous.
func k_projection_async(kx unsafe.Pointer, ky unsafe.Pointer, kz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("projection")
	}

	projection_args.Lock()
	defer projection_args.Unlock()

	if projection_code == 0 {
		projection_code = fatbinLoad(projection_map, "projection")
	}

	projection_args.arg_kx = kx
	projection_args.arg_ky = ky
	projection_args.arg_kz = kz
	projection_args.arg_mx = mx
	projection_args.arg_my = my
	projection_args.arg_mz = mz
	projection_args.arg_N = N

	args := projection_args.argptr[:]
	cu.LaunchKernel(projection_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("projection")
	}
}

// maps compute capability on PTX code for projection kernel.
var projection_map = map[int]string{0: "",
	50: projection_ptx_50}

// projection PTX code for various compute capabilities.
const (
	projection_ptx_50 = `
.version 7.5
.target sm_50
.address_size 64

	// .globl	projection

.visible .entry projection(
	.param .u64 projection_param_0,
	.param .u64 projection_param_1,
	.param .u64 projection_param_2,
	.param .u64 projection_param_3,
	.param .u64 projection_param_4,
	.param .u64 projection_param_5,
	.param .u32 projection_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<20>;


	ld.param.u64 	%rd1, [projection_param_0];
	ld.param.u64 	%rd2, [projection_param_1];
	ld.param.u64 	%rd3, [projection_param_2];
	ld.param.u64 	%rd4, [projection_param_3];
	ld.param.u64 	%rd5, [projection_param_4];
	ld.param.u64 	%rd6, [projection_param_5];
	ld.param.u32 	%r2, [projection_param_6];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd7, %rd1;
	mul.wide.s32 	%rd8, %r1, 4;
	add.s64 	%rd9, %rd7, %rd8;
	cvta.to.global.u64 	%rd10, %rd4;
	add.s64 	%rd11, %rd10, %rd8;
	ld.global.nc.f32 	%f1, [%rd11];
	ld.global.f32 	%f2, [%rd9];
	cvta.to.global.u64 	%rd12, %rd2;
	add.s64 	%rd13, %rd12, %rd8;
	cvta.to.global.u64 	%rd14, %rd5;
	add.s64 	%rd15, %rd14, %rd8;
	ld.global.nc.f32 	%f3, [%rd15];
	ld.global.f32 	%f4, [%rd13];
	mul.f32 	%f5, %f4, %f3;
	fma.rn.f32 	%f6, %f2, %f1, %f5;
	cvta.to.global.u64 	%rd16, %rd3;
	add.s64 	%rd17, %rd16, %rd8;
	cvta.to.global.u64 	%rd18, %rd6;
	add.s64 	%rd19, %rd18, %rd8;
	ld.global.nc.f32 	%f7, [%rd19];
	ld.global.f32 	%f8, [%rd17];
	fma.rn.f32 	%f9, %f8, %f7, %f6;
	mul.f32 	%f10, %f9, %f1;
	sub.f32 	%f11, %f2, %f10;
	st.global.f32 	[%rd9], %f11;
	mul.f32 	%f12, %f9, %f3;
	sub.f32 	%f13, %f4, %f12;
	st.global.f32 	[%rd13], %f13;
	mul.f32 	%f14, %f9, %f7;
	sub.f32 	%f15, %f8, %f14;
	st.global.f32 	[%rd17], %f15;

$L__BB0_2:
	ret;

}

`
)
