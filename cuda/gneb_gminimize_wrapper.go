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

// CUDA handle for gminimize kernel
var gminimize_code cu.Function

// Stores the arguments for gminimize kernel invocation
type gminimize_args_t struct {
	arg_mx unsafe.Pointer
	arg_my unsafe.Pointer
	arg_mz unsafe.Pointer
	arg_Bx unsafe.Pointer
	arg_By unsafe.Pointer
	arg_Bz unsafe.Pointer
	arg_dt float32
	arg_N  int
	argptr [8]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for gminimize kernel invocation
var gminimize_args gminimize_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	gminimize_args.argptr[0] = unsafe.Pointer(&gminimize_args.arg_mx)
	gminimize_args.argptr[1] = unsafe.Pointer(&gminimize_args.arg_my)
	gminimize_args.argptr[2] = unsafe.Pointer(&gminimize_args.arg_mz)
	gminimize_args.argptr[3] = unsafe.Pointer(&gminimize_args.arg_Bx)
	gminimize_args.argptr[4] = unsafe.Pointer(&gminimize_args.arg_By)
	gminimize_args.argptr[5] = unsafe.Pointer(&gminimize_args.arg_Bz)
	gminimize_args.argptr[6] = unsafe.Pointer(&gminimize_args.arg_dt)
	gminimize_args.argptr[7] = unsafe.Pointer(&gminimize_args.arg_N)
}

// Wrapper for gminimize CUDA kernel, asynchronous.
func k_gminimize_async(mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, Bx unsafe.Pointer, By unsafe.Pointer, Bz unsafe.Pointer, dt float32, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("gminimize")
	}

	gminimize_args.Lock()
	defer gminimize_args.Unlock()

	if gminimize_code == 0 {
		gminimize_code = fatbinLoad(gminimize_map, "gminimize")
	}

	gminimize_args.arg_mx = mx
	gminimize_args.arg_my = my
	gminimize_args.arg_mz = mz
	gminimize_args.arg_Bx = Bx
	gminimize_args.arg_By = By
	gminimize_args.arg_Bz = Bz
	gminimize_args.arg_dt = dt
	gminimize_args.arg_N = N

	args := gminimize_args.argptr[:]
	cu.LaunchKernel(gminimize_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("gminimize")
	}
}

// maps compute capability on PTX code for gminimize kernel.
var gminimize_map = map[int]string{0: "",
	50: gminimize_ptx_50}

// gminimize PTX code for various compute capabilities.
const (
	gminimize_ptx_50 = `
.version 7.5
.target sm_50
.address_size 64

	// .globl	gminimize

.visible .entry gminimize(
	.param .u64 gminimize_param_0,
	.param .u64 gminimize_param_1,
	.param .u64 gminimize_param_2,
	.param .u64 gminimize_param_3,
	.param .u64 gminimize_param_4,
	.param .u64 gminimize_param_5,
	.param .f32 gminimize_param_6,
	.param .u32 gminimize_param_7
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<9>;
	.reg .f64 	%fd<41>;
	.reg .b64 	%rd<20>;


	ld.param.u64 	%rd1, [gminimize_param_0];
	ld.param.u64 	%rd2, [gminimize_param_1];
	ld.param.u64 	%rd3, [gminimize_param_2];
	ld.param.u64 	%rd4, [gminimize_param_3];
	ld.param.u64 	%rd5, [gminimize_param_4];
	ld.param.u64 	%rd6, [gminimize_param_5];
	ld.param.f32 	%f1, [gminimize_param_6];
	ld.param.u32 	%r2, [gminimize_param_7];
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
	cvta.to.global.u64 	%rd10, %rd2;
	add.s64 	%rd11, %rd10, %rd8;
	cvta.to.global.u64 	%rd12, %rd3;
	add.s64 	%rd13, %rd12, %rd8;
	cvta.to.global.u64 	%rd14, %rd4;
	add.s64 	%rd15, %rd14, %rd8;
	cvta.to.global.u64 	%rd16, %rd5;
	add.s64 	%rd17, %rd16, %rd8;
	cvta.to.global.u64 	%rd18, %rd6;
	add.s64 	%rd19, %rd18, %rd8;
	ld.global.f32 	%f2, [%rd13];
	setp.lt.f32 	%p2, %f2, 0f00000000;
	selp.f32 	%f3, 0fBF800000, 0f3F800000, %p2;
	ld.global.f32 	%f4, [%rd9];
	cvt.f64.f32 	%fd1, %f4;
	mul.f32 	%f5, %f2, %f3;
	cvt.f64.f32 	%fd2, %f5;
	add.f64 	%fd3, %fd2, 0d3FF0000000000000;
	mov.f64 	%fd4, 0d3FF0000000000000;
	div.rn.f64 	%fd5, %fd1, %fd3;
	cvt.rn.f32.f64 	%f6, %fd5;
	ld.global.f32 	%f7, [%rd11];
	cvt.f64.f32 	%fd6, %f7;
	div.rn.f64 	%fd7, %fd6, %fd3;
	cvt.rn.f32.f64 	%f8, %fd7;
	ld.global.nc.f32 	%f9, [%rd15];
	cvt.f64.f32 	%fd8, %f9;
	mul.f32 	%f10, %f7, %f7;
	cvt.f64.f32 	%fd9, %f10;
	fma.rn.f64 	%fd10, %fd3, %fd2, %fd9;
	mul.f64 	%fd11, %fd10, %fd8;
	ld.global.nc.f32 	%f11, [%rd17];
	mul.f32 	%f12, %f4, %f11;
	mul.f32 	%f13, %f7, %f12;
	cvt.f64.f32 	%fd12, %f13;
	sub.f64 	%fd13, %fd11, %fd12;
	ld.global.nc.f32 	%f14, [%rd19];
	mul.f32 	%f15, %f4, %f14;
	add.f32 	%f16, %f2, %f3;
	mul.f32 	%f17, %f16, %f15;
	cvt.f64.f32 	%fd14, %f17;
	sub.f64 	%fd15, %fd13, %fd14;
	cvt.rn.f32.f64 	%f18, %fd15;
	mul.f32 	%f19, %f4, %f9;
	mul.f32 	%f20, %f7, %f19;
	cvt.f64.f32 	%fd16, %f20;
	cvt.f64.f32 	%fd17, %f11;
	mul.f32 	%f21, %f4, %f4;
	cvt.f64.f32 	%fd18, %f21;
	fma.rn.f64 	%fd19, %fd3, %fd2, %fd18;
	mul.f64 	%fd20, %fd19, %fd17;
	sub.f64 	%fd21, %fd20, %fd16;
	mul.f32 	%f22, %f7, %f14;
	mul.f32 	%f23, %f16, %f22;
	cvt.f64.f32 	%fd22, %f23;
	sub.f64 	%fd23, %fd21, %fd22;
	cvt.rn.f32.f64 	%f24, %fd23;
	fma.rn.f32 	%f25, %f18, %f1, %f6;
	fma.rn.f32 	%f26, %f24, %f1, %f8;
	mul.f32 	%f27, %f25, %f25;
	cvt.f64.f32 	%fd24, %f27;
	add.f64 	%fd25, %fd24, 0d3FF0000000000000;
	mul.f32 	%f28, %f26, %f26;
	cvt.f64.f32 	%fd26, %f28;
	add.f64 	%fd27, %fd25, %fd26;
	rcp.rn.f64 	%fd28, %fd27;
	cvt.rn.f32.f64 	%f29, %fd28;
	cvt.f64.f32 	%fd29, %f25;
	add.f64 	%fd30, %fd29, %fd29;
	cvt.f64.f32 	%fd31, %f29;
	mul.f64 	%fd32, %fd30, %fd31;
	cvt.rn.f32.f64 	%f30, %fd32;
	st.global.f32 	[%rd9], %f30;
	cvt.f64.f32 	%fd33, %f26;
	add.f64 	%fd34, %fd33, %fd33;
	mul.f64 	%fd35, %fd34, %fd31;
	cvt.rn.f32.f64 	%f31, %fd35;
	st.global.f32 	[%rd11], %f31;
	cvt.f64.f32 	%fd36, %f3;
	sub.f64 	%fd37, %fd4, %fd24;
	sub.f64 	%fd38, %fd37, %fd26;
	mul.f64 	%fd39, %fd38, %fd36;
	mul.f64 	%fd40, %fd39, %fd31;
	cvt.rn.f32.f64 	%f32, %fd40;
	st.global.f32 	[%rd13], %f32;

$L__BB0_2:
	ret;

}

`
)