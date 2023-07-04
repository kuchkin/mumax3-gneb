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

// CUDA handle for reducemaxvecdiff2 kernel
var reducemaxvecdiff2_code cu.Function

// Stores the arguments for reducemaxvecdiff2 kernel invocation
type reducemaxvecdiff2_args_t struct {
	arg_x1      unsafe.Pointer
	arg_y1      unsafe.Pointer
	arg_z1      unsafe.Pointer
	arg_x2      unsafe.Pointer
	arg_y2      unsafe.Pointer
	arg_z2      unsafe.Pointer
	arg_dst     unsafe.Pointer
	arg_initVal float32
	arg_n       int
	argptr      [9]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for reducemaxvecdiff2 kernel invocation
var reducemaxvecdiff2_args reducemaxvecdiff2_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	reducemaxvecdiff2_args.argptr[0] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_x1)
	reducemaxvecdiff2_args.argptr[1] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_y1)
	reducemaxvecdiff2_args.argptr[2] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_z1)
	reducemaxvecdiff2_args.argptr[3] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_x2)
	reducemaxvecdiff2_args.argptr[4] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_y2)
	reducemaxvecdiff2_args.argptr[5] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_z2)
	reducemaxvecdiff2_args.argptr[6] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_dst)
	reducemaxvecdiff2_args.argptr[7] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_initVal)
	reducemaxvecdiff2_args.argptr[8] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_n)
}

// Wrapper for reducemaxvecdiff2 CUDA kernel, asynchronous.
func k_reducemaxvecdiff2_async(x1 unsafe.Pointer, y1 unsafe.Pointer, z1 unsafe.Pointer, x2 unsafe.Pointer, y2 unsafe.Pointer, z2 unsafe.Pointer, dst unsafe.Pointer, initVal float32, n int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("reducemaxvecdiff2")
	}

	reducemaxvecdiff2_args.Lock()
	defer reducemaxvecdiff2_args.Unlock()

	if reducemaxvecdiff2_code == 0 {
		reducemaxvecdiff2_code = fatbinLoad(reducemaxvecdiff2_map, "reducemaxvecdiff2")
	}

	reducemaxvecdiff2_args.arg_x1 = x1
	reducemaxvecdiff2_args.arg_y1 = y1
	reducemaxvecdiff2_args.arg_z1 = z1
	reducemaxvecdiff2_args.arg_x2 = x2
	reducemaxvecdiff2_args.arg_y2 = y2
	reducemaxvecdiff2_args.arg_z2 = z2
	reducemaxvecdiff2_args.arg_dst = dst
	reducemaxvecdiff2_args.arg_initVal = initVal
	reducemaxvecdiff2_args.arg_n = n

	args := reducemaxvecdiff2_args.argptr[:]
	cu.LaunchKernel(reducemaxvecdiff2_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("reducemaxvecdiff2")
	}
}

// maps compute capability on PTX code for reducemaxvecdiff2 kernel.
var reducemaxvecdiff2_map = map[int]string{0: "",
	50: reducemaxvecdiff2_ptx_50}

// reducemaxvecdiff2 PTX code for various compute capabilities.
const (
	reducemaxvecdiff2_ptx_50 = `
.version 7.5
.target sm_50
.address_size 64

	// .globl	reducemaxvecdiff2

.visible .entry reducemaxvecdiff2(
	.param .u64 reducemaxvecdiff2_param_0,
	.param .u64 reducemaxvecdiff2_param_1,
	.param .u64 reducemaxvecdiff2_param_2,
	.param .u64 reducemaxvecdiff2_param_3,
	.param .u64 reducemaxvecdiff2_param_4,
	.param .u64 reducemaxvecdiff2_param_5,
	.param .u64 reducemaxvecdiff2_param_6,
	.param .f32 reducemaxvecdiff2_param_7,
	.param .u32 reducemaxvecdiff2_param_8
)
{
	.reg .pred 	%p<11>;
	.reg .f32 	%f<101>;
	.reg .b32 	%r<39>;
	.reg .b64 	%rd<70>;
	// demoted variable
	.shared .align 4 .b8 _ZZ17reducemaxvecdiff2E5sdata[2048];

	ld.param.u64 	%rd31, [reducemaxvecdiff2_param_0];
	ld.param.u64 	%rd27, [reducemaxvecdiff2_param_1];
	ld.param.u64 	%rd32, [reducemaxvecdiff2_param_2];
	ld.param.u64 	%rd28, [reducemaxvecdiff2_param_3];
	ld.param.u64 	%rd29, [reducemaxvecdiff2_param_4];
	ld.param.u64 	%rd33, [reducemaxvecdiff2_param_5];
	ld.param.u64 	%rd30, [reducemaxvecdiff2_param_6];
	ld.param.f32 	%f100, [reducemaxvecdiff2_param_7];
	ld.param.u32 	%r17, [reducemaxvecdiff2_param_8];
	cvta.to.global.u64 	%rd1, %rd33;
	cvta.to.global.u64 	%rd2, %rd32;
	cvta.to.global.u64 	%rd3, %rd31;
	mov.u32 	%r38, %ntid.x;
	mov.u32 	%r18, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r36, %r18, %r38, %r2;
	mov.u32 	%r19, %nctaid.x;
	mul.lo.s32 	%r4, %r19, %r38;
	setp.ge.s32 	%p1, %r36, %r17;
	@%p1 bra 	$L__BB0_7;

	add.s32 	%r20, %r4, %r17;
	add.s32 	%r21, %r36, %r4;
	not.b32 	%r22, %r21;
	add.s32 	%r23, %r20, %r22;
	div.u32 	%r5, %r23, %r4;
	add.s32 	%r24, %r5, 1;
	and.b32  	%r35, %r24, 3;
	setp.eq.s32 	%p2, %r35, 0;
	@%p2 bra 	$L__BB0_4;

	mul.wide.s32 	%rd34, %r36, 4;
	add.s64 	%rd69, %rd1, %rd34;
	mul.wide.s32 	%rd5, %r4, 4;
	add.s64 	%rd68, %rd2, %rd34;
	cvta.to.global.u64 	%rd35, %rd29;
	add.s64 	%rd67, %rd35, %rd34;
	cvta.to.global.u64 	%rd36, %rd27;
	add.s64 	%rd66, %rd36, %rd34;
	cvta.to.global.u64 	%rd37, %rd28;
	add.s64 	%rd65, %rd37, %rd34;
	add.s64 	%rd64, %rd3, %rd34;

$L__BB0_3:
	.pragma "nounroll";
	ld.global.nc.f32 	%f10, [%rd65];
	ld.global.nc.f32 	%f11, [%rd64];
	sub.f32 	%f12, %f11, %f10;
	ld.global.nc.f32 	%f13, [%rd67];
	ld.global.nc.f32 	%f14, [%rd66];
	sub.f32 	%f15, %f14, %f13;
	mul.f32 	%f16, %f15, %f15;
	fma.rn.f32 	%f17, %f12, %f12, %f16;
	ld.global.nc.f32 	%f18, [%rd69];
	ld.global.nc.f32 	%f19, [%rd68];
	sub.f32 	%f20, %f19, %f18;
	fma.rn.f32 	%f21, %f20, %f20, %f17;
	max.f32 	%f100, %f100, %f21;
	add.s32 	%r36, %r36, %r4;
	add.s64 	%rd69, %rd69, %rd5;
	add.s64 	%rd68, %rd68, %rd5;
	add.s64 	%rd67, %rd67, %rd5;
	add.s64 	%rd66, %rd66, %rd5;
	add.s64 	%rd65, %rd65, %rd5;
	add.s64 	%rd64, %rd64, %rd5;
	add.s32 	%r35, %r35, -1;
	setp.ne.s32 	%p3, %r35, 0;
	@%p3 bra 	$L__BB0_3;

$L__BB0_4:
	setp.lt.u32 	%p4, %r5, 3;
	@%p4 bra 	$L__BB0_7;

	mul.wide.s32 	%rd23, %r4, 4;
	cvta.to.global.u64 	%rd24, %rd28;
	cvta.to.global.u64 	%rd25, %rd27;
	cvta.to.global.u64 	%rd26, %rd29;

$L__BB0_6:
	mul.wide.s32 	%rd38, %r36, 4;
	add.s64 	%rd39, %rd3, %rd38;
	add.s64 	%rd40, %rd24, %rd38;
	ld.global.nc.f32 	%f22, [%rd40];
	ld.global.nc.f32 	%f23, [%rd39];
	sub.f32 	%f24, %f23, %f22;
	add.s64 	%rd41, %rd25, %rd38;
	add.s64 	%rd42, %rd26, %rd38;
	ld.global.nc.f32 	%f25, [%rd42];
	ld.global.nc.f32 	%f26, [%rd41];
	sub.f32 	%f27, %f26, %f25;
	mul.f32 	%f28, %f27, %f27;
	fma.rn.f32 	%f29, %f24, %f24, %f28;
	add.s64 	%rd43, %rd2, %rd38;
	add.s64 	%rd44, %rd1, %rd38;
	ld.global.nc.f32 	%f30, [%rd44];
	ld.global.nc.f32 	%f31, [%rd43];
	sub.f32 	%f32, %f31, %f30;
	fma.rn.f32 	%f33, %f32, %f32, %f29;
	max.f32 	%f34, %f100, %f33;
	add.s64 	%rd45, %rd39, %rd23;
	add.s64 	%rd46, %rd40, %rd23;
	ld.global.nc.f32 	%f35, [%rd46];
	ld.global.nc.f32 	%f36, [%rd45];
	sub.f32 	%f37, %f36, %f35;
	add.s64 	%rd47, %rd41, %rd23;
	add.s64 	%rd48, %rd42, %rd23;
	ld.global.nc.f32 	%f38, [%rd48];
	ld.global.nc.f32 	%f39, [%rd47];
	sub.f32 	%f40, %f39, %f38;
	mul.f32 	%f41, %f40, %f40;
	fma.rn.f32 	%f42, %f37, %f37, %f41;
	add.s64 	%rd49, %rd43, %rd23;
	add.s64 	%rd50, %rd44, %rd23;
	ld.global.nc.f32 	%f43, [%rd50];
	ld.global.nc.f32 	%f44, [%rd49];
	sub.f32 	%f45, %f44, %f43;
	fma.rn.f32 	%f46, %f45, %f45, %f42;
	max.f32 	%f47, %f34, %f46;
	add.s32 	%r25, %r36, %r4;
	add.s32 	%r26, %r25, %r4;
	add.s64 	%rd51, %rd45, %rd23;
	add.s64 	%rd52, %rd46, %rd23;
	ld.global.nc.f32 	%f48, [%rd52];
	ld.global.nc.f32 	%f49, [%rd51];
	sub.f32 	%f50, %f49, %f48;
	add.s64 	%rd53, %rd47, %rd23;
	add.s64 	%rd54, %rd48, %rd23;
	ld.global.nc.f32 	%f51, [%rd54];
	ld.global.nc.f32 	%f52, [%rd53];
	sub.f32 	%f53, %f52, %f51;
	mul.f32 	%f54, %f53, %f53;
	fma.rn.f32 	%f55, %f50, %f50, %f54;
	add.s64 	%rd55, %rd49, %rd23;
	add.s64 	%rd56, %rd50, %rd23;
	ld.global.nc.f32 	%f56, [%rd56];
	ld.global.nc.f32 	%f57, [%rd55];
	sub.f32 	%f58, %f57, %f56;
	fma.rn.f32 	%f59, %f58, %f58, %f55;
	max.f32 	%f60, %f47, %f59;
	add.s32 	%r27, %r26, %r4;
	add.s64 	%rd57, %rd51, %rd23;
	add.s64 	%rd58, %rd52, %rd23;
	ld.global.nc.f32 	%f61, [%rd58];
	ld.global.nc.f32 	%f62, [%rd57];
	sub.f32 	%f63, %f62, %f61;
	add.s64 	%rd59, %rd53, %rd23;
	add.s64 	%rd60, %rd54, %rd23;
	ld.global.nc.f32 	%f64, [%rd60];
	ld.global.nc.f32 	%f65, [%rd59];
	sub.f32 	%f66, %f65, %f64;
	mul.f32 	%f67, %f66, %f66;
	fma.rn.f32 	%f68, %f63, %f63, %f67;
	add.s64 	%rd61, %rd55, %rd23;
	add.s64 	%rd62, %rd56, %rd23;
	ld.global.nc.f32 	%f69, [%rd62];
	ld.global.nc.f32 	%f70, [%rd61];
	sub.f32 	%f71, %f70, %f69;
	fma.rn.f32 	%f72, %f71, %f71, %f68;
	max.f32 	%f100, %f60, %f72;
	add.s32 	%r36, %r27, %r4;
	setp.lt.s32 	%p5, %r36, %r17;
	@%p5 bra 	$L__BB0_6;

$L__BB0_7:
	shl.b32 	%r28, %r2, 2;
	mov.u32 	%r29, _ZZ17reducemaxvecdiff2E5sdata;
	add.s32 	%r14, %r29, %r28;
	st.shared.f32 	[%r14], %f100;
	bar.sync 	0;
	setp.lt.u32 	%p6, %r38, 66;
	@%p6 bra 	$L__BB0_12;

$L__BB0_9:
	shr.u32 	%r16, %r38, 1;
	setp.ge.u32 	%p7, %r2, %r16;
	@%p7 bra 	$L__BB0_11;

	ld.shared.f32 	%f73, [%r14];
	shl.b32 	%r30, %r16, 2;
	add.s32 	%r31, %r14, %r30;
	ld.shared.f32 	%f74, [%r31];
	max.f32 	%f75, %f73, %f74;
	st.shared.f32 	[%r14], %f75;

$L__BB0_11:
	bar.sync 	0;
	setp.gt.u32 	%p8, %r38, 131;
	mov.u32 	%r38, %r16;
	@%p8 bra 	$L__BB0_9;

$L__BB0_12:
	setp.gt.s32 	%p9, %r2, 31;
	@%p9 bra 	$L__BB0_14;

	ld.volatile.shared.f32 	%f76, [%r14];
	ld.volatile.shared.f32 	%f77, [%r14+128];
	max.f32 	%f78, %f76, %f77;
	st.volatile.shared.f32 	[%r14], %f78;
	ld.volatile.shared.f32 	%f79, [%r14+64];
	ld.volatile.shared.f32 	%f80, [%r14];
	max.f32 	%f81, %f80, %f79;
	st.volatile.shared.f32 	[%r14], %f81;
	ld.volatile.shared.f32 	%f82, [%r14+32];
	ld.volatile.shared.f32 	%f83, [%r14];
	max.f32 	%f84, %f83, %f82;
	st.volatile.shared.f32 	[%r14], %f84;
	ld.volatile.shared.f32 	%f85, [%r14+16];
	ld.volatile.shared.f32 	%f86, [%r14];
	max.f32 	%f87, %f86, %f85;
	st.volatile.shared.f32 	[%r14], %f87;
	ld.volatile.shared.f32 	%f88, [%r14+8];
	ld.volatile.shared.f32 	%f89, [%r14];
	max.f32 	%f90, %f89, %f88;
	st.volatile.shared.f32 	[%r14], %f90;
	ld.volatile.shared.f32 	%f91, [%r14+4];
	ld.volatile.shared.f32 	%f92, [%r14];
	max.f32 	%f93, %f92, %f91;
	st.volatile.shared.f32 	[%r14], %f93;

$L__BB0_14:
	setp.ne.s32 	%p10, %r2, 0;
	@%p10 bra 	$L__BB0_16;

	ld.shared.f32 	%f94, [_ZZ17reducemaxvecdiff2E5sdata];
	abs.f32 	%f95, %f94;
	cvta.to.global.u64 	%rd63, %rd30;
	mov.b32 	%r32, %f95;
	atom.global.max.s32 	%r33, [%rd63], %r32;

$L__BB0_16:
	ret;

}

`
)