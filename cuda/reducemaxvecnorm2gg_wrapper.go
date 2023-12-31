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

// CUDA handle for reducemaxvecnorm2gg kernel
var reducemaxvecnorm2gg_code cu.Function

// Stores the arguments for reducemaxvecnorm2gg kernel invocation
type reducemaxvecnorm2gg_args_t struct {
	arg_x       unsafe.Pointer
	arg_y       unsafe.Pointer
	arg_z       unsafe.Pointer
	arg_Bx      unsafe.Pointer
	arg_By      unsafe.Pointer
	arg_Bz      unsafe.Pointer
	arg_dst     unsafe.Pointer
	arg_initVal float32
	arg_n       int
	argptr      [9]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for reducemaxvecnorm2gg kernel invocation
var reducemaxvecnorm2gg_args reducemaxvecnorm2gg_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	reducemaxvecnorm2gg_args.argptr[0] = unsafe.Pointer(&reducemaxvecnorm2gg_args.arg_x)
	reducemaxvecnorm2gg_args.argptr[1] = unsafe.Pointer(&reducemaxvecnorm2gg_args.arg_y)
	reducemaxvecnorm2gg_args.argptr[2] = unsafe.Pointer(&reducemaxvecnorm2gg_args.arg_z)
	reducemaxvecnorm2gg_args.argptr[3] = unsafe.Pointer(&reducemaxvecnorm2gg_args.arg_Bx)
	reducemaxvecnorm2gg_args.argptr[4] = unsafe.Pointer(&reducemaxvecnorm2gg_args.arg_By)
	reducemaxvecnorm2gg_args.argptr[5] = unsafe.Pointer(&reducemaxvecnorm2gg_args.arg_Bz)
	reducemaxvecnorm2gg_args.argptr[6] = unsafe.Pointer(&reducemaxvecnorm2gg_args.arg_dst)
	reducemaxvecnorm2gg_args.argptr[7] = unsafe.Pointer(&reducemaxvecnorm2gg_args.arg_initVal)
	reducemaxvecnorm2gg_args.argptr[8] = unsafe.Pointer(&reducemaxvecnorm2gg_args.arg_n)
}

// Wrapper for reducemaxvecnorm2gg CUDA kernel, asynchronous.
func k_reducemaxvecnorm2gg_async(x unsafe.Pointer, y unsafe.Pointer, z unsafe.Pointer, Bx unsafe.Pointer, By unsafe.Pointer, Bz unsafe.Pointer, dst unsafe.Pointer, initVal float32, n int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("reducemaxvecnorm2gg")
	}

	reducemaxvecnorm2gg_args.Lock()
	defer reducemaxvecnorm2gg_args.Unlock()

	if reducemaxvecnorm2gg_code == 0 {
		reducemaxvecnorm2gg_code = fatbinLoad(reducemaxvecnorm2gg_map, "reducemaxvecnorm2gg")
	}

	reducemaxvecnorm2gg_args.arg_x = x
	reducemaxvecnorm2gg_args.arg_y = y
	reducemaxvecnorm2gg_args.arg_z = z
	reducemaxvecnorm2gg_args.arg_Bx = Bx
	reducemaxvecnorm2gg_args.arg_By = By
	reducemaxvecnorm2gg_args.arg_Bz = Bz
	reducemaxvecnorm2gg_args.arg_dst = dst
	reducemaxvecnorm2gg_args.arg_initVal = initVal
	reducemaxvecnorm2gg_args.arg_n = n

	args := reducemaxvecnorm2gg_args.argptr[:]
	cu.LaunchKernel(reducemaxvecnorm2gg_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("reducemaxvecnorm2gg")
	}
}

// maps compute capability on PTX code for reducemaxvecnorm2gg kernel.
var reducemaxvecnorm2gg_map = map[int]string{0: "",
	50: reducemaxvecnorm2gg_ptx_50}

// reducemaxvecnorm2gg PTX code for various compute capabilities.
const (
	reducemaxvecnorm2gg_ptx_50 = `
.version 7.5
.target sm_50
.address_size 64

	// .globl	reducemaxvecnorm2gg

.visible .entry reducemaxvecnorm2gg(
	.param .u64 reducemaxvecnorm2gg_param_0,
	.param .u64 reducemaxvecnorm2gg_param_1,
	.param .u64 reducemaxvecnorm2gg_param_2,
	.param .u64 reducemaxvecnorm2gg_param_3,
	.param .u64 reducemaxvecnorm2gg_param_4,
	.param .u64 reducemaxvecnorm2gg_param_5,
	.param .u64 reducemaxvecnorm2gg_param_6,
	.param .f32 reducemaxvecnorm2gg_param_7,
	.param .u32 reducemaxvecnorm2gg_param_8
)
{
	.reg .pred 	%p<15>;
	.reg .f32 	%f<79>;
	.reg .b32 	%r<30>;
	.reg .f64 	%fd<85>;
	.reg .b64 	%rd<36>;
	// demoted variable
	.shared .align 4 .b8 _ZZ19reducemaxvecnorm2ggE5sdata[2048];

	ld.param.u64 	%rd8, [reducemaxvecnorm2gg_param_0];
	ld.param.u64 	%rd9, [reducemaxvecnorm2gg_param_1];
	ld.param.u64 	%rd10, [reducemaxvecnorm2gg_param_2];
	ld.param.u64 	%rd11, [reducemaxvecnorm2gg_param_3];
	ld.param.u64 	%rd12, [reducemaxvecnorm2gg_param_4];
	ld.param.u64 	%rd13, [reducemaxvecnorm2gg_param_5];
	ld.param.u64 	%rd7, [reducemaxvecnorm2gg_param_6];
	ld.param.f32 	%f78, [reducemaxvecnorm2gg_param_7];
	ld.param.u32 	%r13, [reducemaxvecnorm2gg_param_8];
	cvta.to.global.u64 	%rd1, %rd13;
	cvta.to.global.u64 	%rd2, %rd8;
	cvta.to.global.u64 	%rd3, %rd12;
	cvta.to.global.u64 	%rd4, %rd10;
	cvta.to.global.u64 	%rd5, %rd9;
	cvta.to.global.u64 	%rd6, %rd11;
	mov.u32 	%r29, %ntid.x;
	mov.u32 	%r14, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r28, %r14, %r29, %r2;
	mov.u32 	%r15, %nctaid.x;
	mul.lo.s32 	%r4, %r15, %r29;
	setp.ge.s32 	%p1, %r28, %r13;
	@%p1 bra 	$L__BB0_6;

	add.s32 	%r16, %r4, %r13;
	add.s32 	%r5, %r28, %r4;
	not.b32 	%r17, %r5;
	add.s32 	%r6, %r16, %r17;
	div.u32 	%r18, %r6, %r4;
	and.b32  	%r19, %r18, 1;
	setp.eq.b32 	%p2, %r19, 1;
	mov.pred 	%p3, 0;
	xor.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_3;

	cvt.f64.f32 	%fd1, %f78;
	mul.wide.s32 	%rd14, %r28, 4;
	add.s64 	%rd15, %rd6, %rd14;
	ld.global.nc.f32 	%f9, [%rd15];
	cvt.f64.f32 	%fd2, %f9;
	add.s64 	%rd16, %rd5, %rd14;
	ld.global.nc.f32 	%f10, [%rd16];
	mul.f32 	%f11, %f10, %f10;
	cvt.f64.f32 	%fd3, %f11;
	add.s64 	%rd17, %rd4, %rd14;
	ld.global.nc.f32 	%f12, [%rd17];
	setp.gt.f32 	%p5, %f12, 0f00000000;
	selp.f64 	%fd4, 0d3FF0000000000000, 0dBFF0000000000000, %p5;
	cvt.f64.f32 	%fd5, %f12;
	mul.f64 	%fd6, %fd4, %fd5;
	add.f64 	%fd7, %fd6, 0d3FF0000000000000;
	fma.rn.f64 	%fd8, %fd6, %fd7, %fd3;
	mul.f64 	%fd9, %fd8, %fd2;
	add.s64 	%rd18, %rd3, %rd14;
	add.s64 	%rd19, %rd2, %rd14;
	ld.global.nc.f32 	%f13, [%rd19];
	ld.global.nc.f32 	%f14, [%rd18];
	mul.f32 	%f15, %f14, %f13;
	mul.f32 	%f16, %f15, %f10;
	cvt.f64.f32 	%fd10, %f16;
	sub.f64 	%fd11, %fd9, %fd10;
	add.s64 	%rd20, %rd1, %rd14;
	ld.global.nc.f32 	%f17, [%rd20];
	mul.f32 	%f18, %f17, %f13;
	cvt.f64.f32 	%fd12, %f18;
	add.f64 	%fd13, %fd4, %fd5;
	mul.f64 	%fd14, %fd13, %fd12;
	sub.f64 	%fd15, %fd11, %fd14;
	abs.f64 	%fd16, %fd15;
	mul.f32 	%f19, %f9, %f13;
	mul.f32 	%f20, %f19, %f10;
	cvt.f64.f32 	%fd17, %f20;
	cvt.f64.f32 	%fd18, %f14;
	mul.f32 	%f21, %f13, %f13;
	cvt.f64.f32 	%fd19, %f21;
	fma.rn.f64 	%fd20, %fd6, %fd7, %fd19;
	mul.f64 	%fd21, %fd20, %fd18;
	sub.f64 	%fd22, %fd21, %fd17;
	mul.f32 	%f22, %f17, %f10;
	cvt.f64.f32 	%fd23, %f22;
	mul.f64 	%fd24, %fd13, %fd23;
	sub.f64 	%fd25, %fd22, %fd24;
	abs.f64 	%fd26, %fd25;
	add.f64 	%fd27, %fd16, %fd26;
	max.f64 	%fd28, %fd1, %fd27;
	cvt.rn.f32.f64 	%f78, %fd28;
	mov.u32 	%r28, %r5;

$L__BB0_3:
	setp.gt.u32 	%p6, %r4, %r6;
	@%p6 bra 	$L__BB0_6;

	mul.wide.s32 	%rd28, %r4, 4;

$L__BB0_5:
	mul.wide.s32 	%rd21, %r28, 4;
	add.s64 	%rd22, %rd6, %rd21;
	ld.global.nc.f32 	%f23, [%rd22];
	cvt.f64.f32 	%fd29, %f23;
	add.s64 	%rd23, %rd5, %rd21;
	ld.global.nc.f32 	%f24, [%rd23];
	mul.f32 	%f25, %f24, %f24;
	cvt.f64.f32 	%fd30, %f25;
	add.s64 	%rd24, %rd4, %rd21;
	ld.global.nc.f32 	%f26, [%rd24];
	setp.gt.f32 	%p7, %f26, 0f00000000;
	selp.f64 	%fd31, 0d3FF0000000000000, 0dBFF0000000000000, %p7;
	cvt.f64.f32 	%fd32, %f26;
	mul.f64 	%fd33, %fd31, %fd32;
	add.f64 	%fd34, %fd33, 0d3FF0000000000000;
	fma.rn.f64 	%fd35, %fd33, %fd34, %fd30;
	mul.f64 	%fd36, %fd35, %fd29;
	add.s64 	%rd25, %rd3, %rd21;
	add.s64 	%rd26, %rd2, %rd21;
	ld.global.nc.f32 	%f27, [%rd26];
	ld.global.nc.f32 	%f28, [%rd25];
	mul.f32 	%f29, %f28, %f27;
	mul.f32 	%f30, %f29, %f24;
	cvt.f64.f32 	%fd37, %f30;
	sub.f64 	%fd38, %fd36, %fd37;
	add.s64 	%rd27, %rd1, %rd21;
	ld.global.nc.f32 	%f31, [%rd27];
	mul.f32 	%f32, %f31, %f27;
	cvt.f64.f32 	%fd39, %f32;
	add.f64 	%fd40, %fd31, %fd32;
	mul.f64 	%fd41, %fd40, %fd39;
	sub.f64 	%fd42, %fd38, %fd41;
	abs.f64 	%fd43, %fd42;
	mul.f32 	%f33, %f23, %f27;
	mul.f32 	%f34, %f33, %f24;
	cvt.f64.f32 	%fd44, %f34;
	cvt.f64.f32 	%fd45, %f28;
	mul.f32 	%f35, %f27, %f27;
	cvt.f64.f32 	%fd46, %f35;
	fma.rn.f64 	%fd47, %fd33, %fd34, %fd46;
	mul.f64 	%fd48, %fd47, %fd45;
	sub.f64 	%fd49, %fd48, %fd44;
	mul.f32 	%f36, %f31, %f24;
	cvt.f64.f32 	%fd50, %f36;
	mul.f64 	%fd51, %fd40, %fd50;
	sub.f64 	%fd52, %fd49, %fd51;
	abs.f64 	%fd53, %fd52;
	add.f64 	%fd54, %fd43, %fd53;
	cvt.f64.f32 	%fd55, %f78;
	max.f64 	%fd56, %fd55, %fd54;
	cvt.rn.f32.f64 	%f37, %fd56;
	cvt.f64.f32 	%fd57, %f37;
	add.s64 	%rd29, %rd22, %rd28;
	ld.global.nc.f32 	%f38, [%rd29];
	cvt.f64.f32 	%fd58, %f38;
	add.s64 	%rd30, %rd23, %rd28;
	ld.global.nc.f32 	%f39, [%rd30];
	mul.f32 	%f40, %f39, %f39;
	cvt.f64.f32 	%fd59, %f40;
	add.s64 	%rd31, %rd24, %rd28;
	ld.global.nc.f32 	%f41, [%rd31];
	setp.gt.f32 	%p8, %f41, 0f00000000;
	selp.f64 	%fd60, 0d3FF0000000000000, 0dBFF0000000000000, %p8;
	cvt.f64.f32 	%fd61, %f41;
	mul.f64 	%fd62, %fd60, %fd61;
	add.f64 	%fd63, %fd62, 0d3FF0000000000000;
	fma.rn.f64 	%fd64, %fd62, %fd63, %fd59;
	mul.f64 	%fd65, %fd64, %fd58;
	add.s64 	%rd32, %rd25, %rd28;
	add.s64 	%rd33, %rd26, %rd28;
	ld.global.nc.f32 	%f42, [%rd33];
	ld.global.nc.f32 	%f43, [%rd32];
	mul.f32 	%f44, %f43, %f42;
	mul.f32 	%f45, %f44, %f39;
	cvt.f64.f32 	%fd66, %f45;
	sub.f64 	%fd67, %fd65, %fd66;
	add.s64 	%rd34, %rd27, %rd28;
	ld.global.nc.f32 	%f46, [%rd34];
	mul.f32 	%f47, %f46, %f42;
	cvt.f64.f32 	%fd68, %f47;
	add.f64 	%fd69, %fd60, %fd61;
	mul.f64 	%fd70, %fd69, %fd68;
	sub.f64 	%fd71, %fd67, %fd70;
	abs.f64 	%fd72, %fd71;
	mul.f32 	%f48, %f38, %f42;
	mul.f32 	%f49, %f48, %f39;
	cvt.f64.f32 	%fd73, %f49;
	cvt.f64.f32 	%fd74, %f43;
	mul.f32 	%f50, %f42, %f42;
	cvt.f64.f32 	%fd75, %f50;
	fma.rn.f64 	%fd76, %fd62, %fd63, %fd75;
	mul.f64 	%fd77, %fd76, %fd74;
	sub.f64 	%fd78, %fd77, %fd73;
	mul.f32 	%f51, %f46, %f39;
	cvt.f64.f32 	%fd79, %f51;
	mul.f64 	%fd80, %fd69, %fd79;
	sub.f64 	%fd81, %fd78, %fd80;
	abs.f64 	%fd82, %fd81;
	add.f64 	%fd83, %fd72, %fd82;
	max.f64 	%fd84, %fd57, %fd83;
	cvt.rn.f32.f64 	%f78, %fd84;
	add.s32 	%r20, %r28, %r4;
	add.s32 	%r28, %r20, %r4;
	setp.lt.s32 	%p9, %r28, %r13;
	@%p9 bra 	$L__BB0_5;

$L__BB0_6:
	shl.b32 	%r21, %r2, 2;
	mov.u32 	%r22, _ZZ19reducemaxvecnorm2ggE5sdata;
	add.s32 	%r10, %r22, %r21;
	st.shared.f32 	[%r10], %f78;
	bar.sync 	0;
	setp.lt.u32 	%p10, %r29, 66;
	@%p10 bra 	$L__BB0_11;

$L__BB0_8:
	shr.u32 	%r12, %r29, 1;
	setp.ge.u32 	%p11, %r2, %r12;
	@%p11 bra 	$L__BB0_10;

	ld.shared.f32 	%f52, [%r10];
	shl.b32 	%r23, %r12, 2;
	add.s32 	%r24, %r10, %r23;
	ld.shared.f32 	%f53, [%r24];
	max.f32 	%f54, %f52, %f53;
	st.shared.f32 	[%r10], %f54;

$L__BB0_10:
	bar.sync 	0;
	setp.gt.u32 	%p12, %r29, 131;
	mov.u32 	%r29, %r12;
	@%p12 bra 	$L__BB0_8;

$L__BB0_11:
	setp.gt.s32 	%p13, %r2, 31;
	@%p13 bra 	$L__BB0_13;

	ld.volatile.shared.f32 	%f55, [%r10];
	ld.volatile.shared.f32 	%f56, [%r10+128];
	max.f32 	%f57, %f55, %f56;
	st.volatile.shared.f32 	[%r10], %f57;
	ld.volatile.shared.f32 	%f58, [%r10+64];
	ld.volatile.shared.f32 	%f59, [%r10];
	max.f32 	%f60, %f59, %f58;
	st.volatile.shared.f32 	[%r10], %f60;
	ld.volatile.shared.f32 	%f61, [%r10+32];
	ld.volatile.shared.f32 	%f62, [%r10];
	max.f32 	%f63, %f62, %f61;
	st.volatile.shared.f32 	[%r10], %f63;
	ld.volatile.shared.f32 	%f64, [%r10+16];
	ld.volatile.shared.f32 	%f65, [%r10];
	max.f32 	%f66, %f65, %f64;
	st.volatile.shared.f32 	[%r10], %f66;
	ld.volatile.shared.f32 	%f67, [%r10+8];
	ld.volatile.shared.f32 	%f68, [%r10];
	max.f32 	%f69, %f68, %f67;
	st.volatile.shared.f32 	[%r10], %f69;
	ld.volatile.shared.f32 	%f70, [%r10+4];
	ld.volatile.shared.f32 	%f71, [%r10];
	max.f32 	%f72, %f71, %f70;
	st.volatile.shared.f32 	[%r10], %f72;

$L__BB0_13:
	setp.ne.s32 	%p14, %r2, 0;
	@%p14 bra 	$L__BB0_15;

	ld.shared.f32 	%f73, [_ZZ19reducemaxvecnorm2ggE5sdata];
	abs.f32 	%f74, %f73;
	mov.b32 	%r25, %f74;
	cvta.to.global.u64 	%rd35, %rd7;
	atom.global.max.s32 	%r26, [%rd35], %r25;

$L__BB0_15:
	ret;

}

`
)
