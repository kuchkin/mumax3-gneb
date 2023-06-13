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

// CUDA handle for setmaxangle kernel
var setmaxangle_code cu.Function

// Stores the arguments for setmaxangle kernel invocation
type setmaxangle_args_t struct {
	arg_dst     unsafe.Pointer
	arg_mx      unsafe.Pointer
	arg_my      unsafe.Pointer
	arg_mz      unsafe.Pointer
	arg_aLUT2d  unsafe.Pointer
	arg_regions unsafe.Pointer
	arg_Nx      int
	arg_Ny      int
	arg_Nz      int
	arg_PBC     byte
	argptr      [10]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for setmaxangle kernel invocation
var setmaxangle_args setmaxangle_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	setmaxangle_args.argptr[0] = unsafe.Pointer(&setmaxangle_args.arg_dst)
	setmaxangle_args.argptr[1] = unsafe.Pointer(&setmaxangle_args.arg_mx)
	setmaxangle_args.argptr[2] = unsafe.Pointer(&setmaxangle_args.arg_my)
	setmaxangle_args.argptr[3] = unsafe.Pointer(&setmaxangle_args.arg_mz)
	setmaxangle_args.argptr[4] = unsafe.Pointer(&setmaxangle_args.arg_aLUT2d)
	setmaxangle_args.argptr[5] = unsafe.Pointer(&setmaxangle_args.arg_regions)
	setmaxangle_args.argptr[6] = unsafe.Pointer(&setmaxangle_args.arg_Nx)
	setmaxangle_args.argptr[7] = unsafe.Pointer(&setmaxangle_args.arg_Ny)
	setmaxangle_args.argptr[8] = unsafe.Pointer(&setmaxangle_args.arg_Nz)
	setmaxangle_args.argptr[9] = unsafe.Pointer(&setmaxangle_args.arg_PBC)
}

// Wrapper for setmaxangle CUDA kernel, asynchronous.
func k_setmaxangle_async(dst unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, aLUT2d unsafe.Pointer, regions unsafe.Pointer, Nx int, Ny int, Nz int, PBC byte, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("setmaxangle")
	}

	setmaxangle_args.Lock()
	defer setmaxangle_args.Unlock()

	if setmaxangle_code == 0 {
		setmaxangle_code = fatbinLoad(setmaxangle_map, "setmaxangle")
	}

	setmaxangle_args.arg_dst = dst
	setmaxangle_args.arg_mx = mx
	setmaxangle_args.arg_my = my
	setmaxangle_args.arg_mz = mz
	setmaxangle_args.arg_aLUT2d = aLUT2d
	setmaxangle_args.arg_regions = regions
	setmaxangle_args.arg_Nx = Nx
	setmaxangle_args.arg_Ny = Ny
	setmaxangle_args.arg_Nz = Nz
	setmaxangle_args.arg_PBC = PBC

	args := setmaxangle_args.argptr[:]
	cu.LaunchKernel(setmaxangle_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("setmaxangle")
	}
}

// maps compute capability on PTX code for setmaxangle kernel.
var setmaxangle_map = map[int]string{0: "",
	50: setmaxangle_ptx_50}

// setmaxangle PTX code for various compute capabilities.
const (
	setmaxangle_ptx_50 = `
.version 7.5
.target sm_50
.address_size 64

	// .globl	setmaxangle

.visible .entry setmaxangle(
	.param .u64 setmaxangle_param_0,
	.param .u64 setmaxangle_param_1,
	.param .u64 setmaxangle_param_2,
	.param .u64 setmaxangle_param_3,
	.param .u64 setmaxangle_param_4,
	.param .u64 setmaxangle_param_5,
	.param .u32 setmaxangle_param_6,
	.param .u32 setmaxangle_param_7,
	.param .u32 setmaxangle_param_8,
	.param .u8 setmaxangle_param_9
)
{
	.reg .pred 	%p<38>;
	.reg .b16 	%rs<36>;
	.reg .f32 	%f<258>;
	.reg .b32 	%r<111>;
	.reg .b64 	%rd<69>;


	ld.param.u8 	%rs5, [setmaxangle_param_9];
	ld.param.u64 	%rd6, [setmaxangle_param_0];
	ld.param.u64 	%rd7, [setmaxangle_param_1];
	ld.param.u64 	%rd8, [setmaxangle_param_2];
	ld.param.u64 	%rd9, [setmaxangle_param_3];
	ld.param.u64 	%rd10, [setmaxangle_param_4];
	ld.param.u64 	%rd11, [setmaxangle_param_5];
	ld.param.u32 	%r31, [setmaxangle_param_6];
	ld.param.u32 	%r32, [setmaxangle_param_7];
	ld.param.u32 	%r33, [setmaxangle_param_8];
	cvta.to.global.u64 	%rd1, %rd10;
	cvta.to.global.u64 	%rd2, %rd11;
	cvta.to.global.u64 	%rd3, %rd9;
	cvta.to.global.u64 	%rd4, %rd8;
	cvta.to.global.u64 	%rd5, %rd7;
	mov.u32 	%r34, %ntid.x;
	mov.u32 	%r35, %ctaid.x;
	mov.u32 	%r36, %tid.x;
	mad.lo.s32 	%r1, %r35, %r34, %r36;
	mov.u32 	%r37, %ntid.y;
	mov.u32 	%r38, %ctaid.y;
	mov.u32 	%r39, %tid.y;
	mad.lo.s32 	%r2, %r38, %r37, %r39;
	mov.u32 	%r40, %ntid.z;
	mov.u32 	%r41, %ctaid.z;
	mov.u32 	%r42, %tid.z;
	mad.lo.s32 	%r3, %r41, %r40, %r42;
	setp.ge.s32 	%p1, %r1, %r31;
	setp.ge.s32 	%p2, %r2, %r32;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r33;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_34;

	mul.lo.s32 	%r4, %r3, %r32;
	add.s32 	%r43, %r4, %r2;
	mul.lo.s32 	%r5, %r43, %r31;
	add.s32 	%r6, %r5, %r1;
	mul.wide.s32 	%rd12, %r6, 4;
	add.s64 	%rd13, %rd5, %rd12;
	add.s64 	%rd14, %rd4, %rd12;
	add.s64 	%rd15, %rd3, %rd12;
	ld.global.nc.f32 	%f1, [%rd13];
	ld.global.nc.f32 	%f2, [%rd14];
	ld.global.nc.f32 	%f3, [%rd15];
	mul.f32 	%f37, %f2, %f2;
	fma.rn.f32 	%f38, %f1, %f1, %f37;
	fma.rn.f32 	%f39, %f3, %f3, %f38;
	setp.eq.f32 	%p6, %f39, 0f00000000;
	@%p6 bra 	$L__BB0_34;

	cvt.s64.s32 	%rd16, %r6;
	add.s64 	%rd17, %rd2, %rd16;
	ld.global.nc.u8 	%rs1, [%rd17];
	and.b16  	%rs2, %rs5, 1;
	setp.eq.s16 	%p7, %rs2, 0;
	add.s32 	%r7, %r1, -1;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	max.s32 	%r105, %r7, 0;
	bra.uni 	$L__BB0_5;

$L__BB0_3:
	rem.s32 	%r44, %r7, %r31;
	add.s32 	%r45, %r44, %r31;
	rem.s32 	%r105, %r45, %r31;

$L__BB0_5:
	add.s32 	%r46, %r105, %r5;
	cvt.s64.s32 	%rd18, %r46;
	mul.wide.s32 	%rd19, %r46, 4;
	add.s64 	%rd20, %rd5, %rd19;
	add.s64 	%rd21, %rd4, %rd19;
	add.s64 	%rd22, %rd3, %rd19;
	ld.global.nc.f32 	%f41, [%rd22];
	ld.global.nc.f32 	%f42, [%rd20];
	ld.global.nc.f32 	%f43, [%rd21];
	mul.f32 	%f44, %f43, %f43;
	fma.rn.f32 	%f45, %f42, %f42, %f44;
	fma.rn.f32 	%f46, %f41, %f41, %f45;
	setp.eq.f32 	%p8, %f46, 0f00000000;
	mov.f32 	%f253, 0f00000000;
	selp.f32 	%f9, %f3, %f41, %p8;
	selp.f32 	%f8, %f2, %f43, %p8;
	selp.f32 	%f7, %f1, %f42, %p8;
	add.s64 	%rd23, %rd2, %rd18;
	ld.global.nc.u8 	%rs6, [%rd23];
	min.u16 	%rs9, %rs6, %rs1;
	cvt.u32.u16 	%r47, %rs9;
	max.u16 	%rs10, %rs6, %rs1;
	cvt.u32.u16 	%r48, %rs10;
	add.s32 	%r49, %r48, 1;
	mul.lo.s32 	%r50, %r49, %r48;
	shr.u32 	%r51, %r50, 1;
	add.s32 	%r52, %r51, %r47;
	mul.wide.s32 	%rd24, %r52, 4;
	add.s64 	%rd25, %rd1, %rd24;
	ld.global.nc.f32 	%f47, [%rd25];
	setp.eq.f32 	%p9, %f47, 0f00000000;
	@%p9 bra 	$L__BB0_7;

	mul.f32 	%f48, %f2, %f8;
	fma.rn.f32 	%f49, %f1, %f7, %f48;
	fma.rn.f32 	%f50, %f3, %f9, %f49;
	abs.f32 	%f51, %f50;
	mov.f32 	%f52, 0f3F800000;
	sub.f32 	%f53, %f52, %f51;
	mul.f32 	%f54, %f53, 0f3F000000;
	sqrt.rn.f32 	%f55, %f54;
	setp.gt.f32 	%p10, %f51, 0f3F11EB85;
	selp.f32 	%f56, %f55, %f51, %p10;
	mul.f32 	%f57, %f56, %f56;
	mov.f32 	%f58, 0f3C94D2E9;
	mov.f32 	%f59, 0f3D53F941;
	fma.rn.f32 	%f60, %f59, %f57, %f58;
	mov.f32 	%f61, 0f3D3F841F;
	fma.rn.f32 	%f62, %f60, %f57, %f61;
	mov.f32 	%f63, 0f3D994929;
	fma.rn.f32 	%f64, %f62, %f57, %f63;
	mov.f32 	%f65, 0f3E2AAB94;
	fma.rn.f32 	%f66, %f64, %f57, %f65;
	mul.f32 	%f67, %f57, %f66;
	fma.rn.f32 	%f68, %f67, %f56, %f56;
	add.f32 	%f69, %f68, %f68;
	mov.f32 	%f70, 0f3FC90FDB;
	sub.f32 	%f71, %f70, %f68;
	selp.f32 	%f72, %f69, %f71, %p10;
	setp.lt.f32 	%p11, %f50, 0f00000000;
	mov.f32 	%f73, 0f00000000;
	mov.f32 	%f74, 0f40490FDB;
	sub.f32 	%f75, %f74, %f72;
	selp.f32 	%f76, %f75, %f72, %p11;
	max.f32 	%f253, %f73, %f76;

$L__BB0_7:
	add.s32 	%r11, %r1, 1;
	@%p7 bra 	$L__BB0_9;
	bra.uni 	$L__BB0_8;

$L__BB0_9:
	add.s32 	%r55, %r31, -1;
	min.s32 	%r106, %r11, %r55;
	bra.uni 	$L__BB0_10;

$L__BB0_8:
	rem.s32 	%r53, %r11, %r31;
	add.s32 	%r54, %r53, %r31;
	rem.s32 	%r106, %r54, %r31;

$L__BB0_10:
	add.s32 	%r56, %r106, %r5;
	cvt.s64.s32 	%rd26, %r56;
	mul.wide.s32 	%rd27, %r56, 4;
	add.s64 	%rd28, %rd5, %rd27;
	add.s64 	%rd29, %rd4, %rd27;
	add.s64 	%rd30, %rd3, %rd27;
	ld.global.nc.f32 	%f77, [%rd30];
	ld.global.nc.f32 	%f78, [%rd28];
	ld.global.nc.f32 	%f79, [%rd29];
	mul.f32 	%f80, %f79, %f79;
	fma.rn.f32 	%f81, %f78, %f78, %f80;
	fma.rn.f32 	%f82, %f77, %f77, %f81;
	setp.eq.f32 	%p13, %f82, 0f00000000;
	selp.f32 	%f14, %f3, %f77, %p13;
	selp.f32 	%f13, %f2, %f79, %p13;
	selp.f32 	%f12, %f1, %f78, %p13;
	add.s64 	%rd31, %rd2, %rd26;
	ld.global.nc.u8 	%rs11, [%rd31];
	min.u16 	%rs14, %rs11, %rs1;
	cvt.u32.u16 	%r57, %rs14;
	max.u16 	%rs15, %rs11, %rs1;
	cvt.u32.u16 	%r58, %rs15;
	add.s32 	%r59, %r58, 1;
	mul.lo.s32 	%r60, %r59, %r58;
	shr.u32 	%r61, %r60, 1;
	add.s32 	%r62, %r61, %r57;
	mul.wide.s32 	%rd32, %r62, 4;
	add.s64 	%rd33, %rd1, %rd32;
	ld.global.nc.f32 	%f83, [%rd33];
	setp.eq.f32 	%p14, %f83, 0f00000000;
	@%p14 bra 	$L__BB0_12;

	mul.f32 	%f84, %f2, %f13;
	fma.rn.f32 	%f85, %f1, %f12, %f84;
	fma.rn.f32 	%f86, %f3, %f14, %f85;
	abs.f32 	%f87, %f86;
	mov.f32 	%f88, 0f3F800000;
	sub.f32 	%f89, %f88, %f87;
	mul.f32 	%f90, %f89, 0f3F000000;
	sqrt.rn.f32 	%f91, %f90;
	setp.gt.f32 	%p15, %f87, 0f3F11EB85;
	selp.f32 	%f92, %f91, %f87, %p15;
	mul.f32 	%f93, %f92, %f92;
	mov.f32 	%f94, 0f3C94D2E9;
	mov.f32 	%f95, 0f3D53F941;
	fma.rn.f32 	%f96, %f95, %f93, %f94;
	mov.f32 	%f97, 0f3D3F841F;
	fma.rn.f32 	%f98, %f96, %f93, %f97;
	mov.f32 	%f99, 0f3D994929;
	fma.rn.f32 	%f100, %f98, %f93, %f99;
	mov.f32 	%f101, 0f3E2AAB94;
	fma.rn.f32 	%f102, %f100, %f93, %f101;
	mul.f32 	%f103, %f93, %f102;
	fma.rn.f32 	%f104, %f103, %f92, %f92;
	add.f32 	%f105, %f104, %f104;
	mov.f32 	%f106, 0f3FC90FDB;
	sub.f32 	%f107, %f106, %f104;
	selp.f32 	%f108, %f105, %f107, %p15;
	setp.lt.f32 	%p16, %f86, 0f00000000;
	mov.f32 	%f109, 0f40490FDB;
	sub.f32 	%f110, %f109, %f108;
	selp.f32 	%f111, %f110, %f108, %p16;
	max.f32 	%f253, %f253, %f111;

$L__BB0_12:
	and.b16  	%rs3, %rs5, 2;
	setp.eq.s16 	%p17, %rs3, 0;
	add.s32 	%r15, %r2, -1;
	@%p17 bra 	$L__BB0_14;
	bra.uni 	$L__BB0_13;

$L__BB0_14:
	max.s32 	%r107, %r15, 0;
	bra.uni 	$L__BB0_15;

$L__BB0_13:
	rem.s32 	%r63, %r15, %r32;
	add.s32 	%r64, %r63, %r32;
	rem.s32 	%r107, %r64, %r32;

$L__BB0_15:
	add.s32 	%r65, %r107, %r4;
	mad.lo.s32 	%r66, %r65, %r31, %r1;
	cvt.s64.s32 	%rd34, %r66;
	mul.wide.s32 	%rd35, %r66, 4;
	add.s64 	%rd36, %rd5, %rd35;
	add.s64 	%rd37, %rd4, %rd35;
	add.s64 	%rd38, %rd3, %rd35;
	ld.global.nc.f32 	%f112, [%rd38];
	ld.global.nc.f32 	%f113, [%rd36];
	ld.global.nc.f32 	%f114, [%rd37];
	mul.f32 	%f115, %f114, %f114;
	fma.rn.f32 	%f116, %f113, %f113, %f115;
	fma.rn.f32 	%f117, %f112, %f112, %f116;
	setp.eq.f32 	%p18, %f117, 0f00000000;
	selp.f32 	%f19, %f3, %f112, %p18;
	selp.f32 	%f18, %f2, %f114, %p18;
	selp.f32 	%f17, %f1, %f113, %p18;
	add.s64 	%rd39, %rd2, %rd34;
	ld.global.nc.u8 	%rs16, [%rd39];
	min.u16 	%rs19, %rs16, %rs1;
	cvt.u32.u16 	%r67, %rs19;
	max.u16 	%rs20, %rs16, %rs1;
	cvt.u32.u16 	%r68, %rs20;
	add.s32 	%r69, %r68, 1;
	mul.lo.s32 	%r70, %r69, %r68;
	shr.u32 	%r71, %r70, 1;
	add.s32 	%r72, %r71, %r67;
	mul.wide.s32 	%rd40, %r72, 4;
	add.s64 	%rd41, %rd1, %rd40;
	ld.global.nc.f32 	%f118, [%rd41];
	setp.eq.f32 	%p19, %f118, 0f00000000;
	@%p19 bra 	$L__BB0_17;

	mul.f32 	%f119, %f2, %f18;
	fma.rn.f32 	%f120, %f1, %f17, %f119;
	fma.rn.f32 	%f121, %f3, %f19, %f120;
	abs.f32 	%f122, %f121;
	mov.f32 	%f123, 0f3F800000;
	sub.f32 	%f124, %f123, %f122;
	mul.f32 	%f125, %f124, 0f3F000000;
	sqrt.rn.f32 	%f126, %f125;
	setp.gt.f32 	%p20, %f122, 0f3F11EB85;
	selp.f32 	%f127, %f126, %f122, %p20;
	mul.f32 	%f128, %f127, %f127;
	mov.f32 	%f129, 0f3C94D2E9;
	mov.f32 	%f130, 0f3D53F941;
	fma.rn.f32 	%f131, %f130, %f128, %f129;
	mov.f32 	%f132, 0f3D3F841F;
	fma.rn.f32 	%f133, %f131, %f128, %f132;
	mov.f32 	%f134, 0f3D994929;
	fma.rn.f32 	%f135, %f133, %f128, %f134;
	mov.f32 	%f136, 0f3E2AAB94;
	fma.rn.f32 	%f137, %f135, %f128, %f136;
	mul.f32 	%f138, %f128, %f137;
	fma.rn.f32 	%f139, %f138, %f127, %f127;
	add.f32 	%f140, %f139, %f139;
	mov.f32 	%f141, 0f3FC90FDB;
	sub.f32 	%f142, %f141, %f139;
	selp.f32 	%f143, %f140, %f142, %p20;
	setp.lt.f32 	%p21, %f121, 0f00000000;
	mov.f32 	%f144, 0f40490FDB;
	sub.f32 	%f145, %f144, %f143;
	selp.f32 	%f146, %f145, %f143, %p21;
	max.f32 	%f253, %f253, %f146;

$L__BB0_17:
	add.s32 	%r19, %r2, 1;
	@%p17 bra 	$L__BB0_19;
	bra.uni 	$L__BB0_18;

$L__BB0_19:
	add.s32 	%r75, %r32, -1;
	min.s32 	%r108, %r19, %r75;
	bra.uni 	$L__BB0_20;

$L__BB0_18:
	rem.s32 	%r73, %r19, %r32;
	add.s32 	%r74, %r73, %r32;
	rem.s32 	%r108, %r74, %r32;

$L__BB0_20:
	add.s32 	%r76, %r108, %r4;
	mad.lo.s32 	%r77, %r76, %r31, %r1;
	cvt.s64.s32 	%rd42, %r77;
	mul.wide.s32 	%rd43, %r77, 4;
	add.s64 	%rd44, %rd5, %rd43;
	add.s64 	%rd45, %rd4, %rd43;
	add.s64 	%rd46, %rd3, %rd43;
	ld.global.nc.f32 	%f147, [%rd46];
	ld.global.nc.f32 	%f148, [%rd44];
	ld.global.nc.f32 	%f149, [%rd45];
	mul.f32 	%f150, %f149, %f149;
	fma.rn.f32 	%f151, %f148, %f148, %f150;
	fma.rn.f32 	%f152, %f147, %f147, %f151;
	setp.eq.f32 	%p23, %f152, 0f00000000;
	selp.f32 	%f24, %f3, %f147, %p23;
	selp.f32 	%f23, %f2, %f149, %p23;
	selp.f32 	%f22, %f1, %f148, %p23;
	add.s64 	%rd47, %rd2, %rd42;
	ld.global.nc.u8 	%rs21, [%rd47];
	min.u16 	%rs24, %rs21, %rs1;
	cvt.u32.u16 	%r78, %rs24;
	max.u16 	%rs25, %rs21, %rs1;
	cvt.u32.u16 	%r79, %rs25;
	add.s32 	%r80, %r79, 1;
	mul.lo.s32 	%r81, %r80, %r79;
	shr.u32 	%r82, %r81, 1;
	add.s32 	%r83, %r82, %r78;
	mul.wide.s32 	%rd48, %r83, 4;
	add.s64 	%rd49, %rd1, %rd48;
	ld.global.nc.f32 	%f153, [%rd49];
	setp.eq.f32 	%p24, %f153, 0f00000000;
	@%p24 bra 	$L__BB0_22;

	mul.f32 	%f154, %f2, %f23;
	fma.rn.f32 	%f155, %f1, %f22, %f154;
	fma.rn.f32 	%f156, %f3, %f24, %f155;
	abs.f32 	%f157, %f156;
	mov.f32 	%f158, 0f3F800000;
	sub.f32 	%f159, %f158, %f157;
	mul.f32 	%f160, %f159, 0f3F000000;
	sqrt.rn.f32 	%f161, %f160;
	setp.gt.f32 	%p25, %f157, 0f3F11EB85;
	selp.f32 	%f162, %f161, %f157, %p25;
	mul.f32 	%f163, %f162, %f162;
	mov.f32 	%f164, 0f3C94D2E9;
	mov.f32 	%f165, 0f3D53F941;
	fma.rn.f32 	%f166, %f165, %f163, %f164;
	mov.f32 	%f167, 0f3D3F841F;
	fma.rn.f32 	%f168, %f166, %f163, %f167;
	mov.f32 	%f169, 0f3D994929;
	fma.rn.f32 	%f170, %f168, %f163, %f169;
	mov.f32 	%f171, 0f3E2AAB94;
	fma.rn.f32 	%f172, %f170, %f163, %f171;
	mul.f32 	%f173, %f163, %f172;
	fma.rn.f32 	%f174, %f173, %f162, %f162;
	add.f32 	%f175, %f174, %f174;
	mov.f32 	%f176, 0f3FC90FDB;
	sub.f32 	%f177, %f176, %f174;
	selp.f32 	%f178, %f175, %f177, %p25;
	setp.lt.f32 	%p26, %f156, 0f00000000;
	mov.f32 	%f179, 0f40490FDB;
	sub.f32 	%f180, %f179, %f178;
	selp.f32 	%f181, %f180, %f178, %p26;
	max.f32 	%f253, %f253, %f181;

$L__BB0_22:
	setp.eq.s32 	%p27, %r33, 1;
	@%p27 bra 	$L__BB0_33;

	and.b16  	%rs4, %rs5, 4;
	setp.eq.s16 	%p28, %rs4, 0;
	add.s32 	%r23, %r3, -1;
	@%p28 bra 	$L__BB0_25;
	bra.uni 	$L__BB0_24;

$L__BB0_25:
	max.s32 	%r109, %r23, 0;
	bra.uni 	$L__BB0_26;

$L__BB0_24:
	rem.s32 	%r84, %r23, %r33;
	add.s32 	%r85, %r84, %r33;
	rem.s32 	%r109, %r85, %r33;

$L__BB0_26:
	mad.lo.s32 	%r86, %r109, %r32, %r2;
	mad.lo.s32 	%r87, %r86, %r31, %r1;
	cvt.s64.s32 	%rd50, %r87;
	mul.wide.s32 	%rd51, %r87, 4;
	add.s64 	%rd52, %rd5, %rd51;
	add.s64 	%rd53, %rd4, %rd51;
	add.s64 	%rd54, %rd3, %rd51;
	ld.global.nc.f32 	%f182, [%rd54];
	ld.global.nc.f32 	%f183, [%rd52];
	ld.global.nc.f32 	%f184, [%rd53];
	mul.f32 	%f185, %f184, %f184;
	fma.rn.f32 	%f186, %f183, %f183, %f185;
	fma.rn.f32 	%f187, %f182, %f182, %f186;
	setp.eq.f32 	%p29, %f187, 0f00000000;
	selp.f32 	%f29, %f3, %f182, %p29;
	selp.f32 	%f28, %f2, %f184, %p29;
	selp.f32 	%f27, %f1, %f183, %p29;
	add.s64 	%rd55, %rd2, %rd50;
	ld.global.nc.u8 	%rs26, [%rd55];
	min.u16 	%rs29, %rs26, %rs1;
	cvt.u32.u16 	%r88, %rs29;
	max.u16 	%rs30, %rs26, %rs1;
	cvt.u32.u16 	%r89, %rs30;
	add.s32 	%r90, %r89, 1;
	mul.lo.s32 	%r91, %r90, %r89;
	shr.u32 	%r92, %r91, 1;
	add.s32 	%r93, %r92, %r88;
	mul.wide.s32 	%rd56, %r93, 4;
	add.s64 	%rd57, %rd1, %rd56;
	ld.global.nc.f32 	%f188, [%rd57];
	setp.eq.f32 	%p30, %f188, 0f00000000;
	@%p30 bra 	$L__BB0_28;

	mul.f32 	%f189, %f2, %f28;
	fma.rn.f32 	%f190, %f1, %f27, %f189;
	fma.rn.f32 	%f191, %f3, %f29, %f190;
	abs.f32 	%f192, %f191;
	mov.f32 	%f193, 0f3F800000;
	sub.f32 	%f194, %f193, %f192;
	mul.f32 	%f195, %f194, 0f3F000000;
	sqrt.rn.f32 	%f196, %f195;
	setp.gt.f32 	%p31, %f192, 0f3F11EB85;
	selp.f32 	%f197, %f196, %f192, %p31;
	mul.f32 	%f198, %f197, %f197;
	mov.f32 	%f199, 0f3C94D2E9;
	mov.f32 	%f200, 0f3D53F941;
	fma.rn.f32 	%f201, %f200, %f198, %f199;
	mov.f32 	%f202, 0f3D3F841F;
	fma.rn.f32 	%f203, %f201, %f198, %f202;
	mov.f32 	%f204, 0f3D994929;
	fma.rn.f32 	%f205, %f203, %f198, %f204;
	mov.f32 	%f206, 0f3E2AAB94;
	fma.rn.f32 	%f207, %f205, %f198, %f206;
	mul.f32 	%f208, %f198, %f207;
	fma.rn.f32 	%f209, %f208, %f197, %f197;
	add.f32 	%f210, %f209, %f209;
	mov.f32 	%f211, 0f3FC90FDB;
	sub.f32 	%f212, %f211, %f209;
	selp.f32 	%f213, %f210, %f212, %p31;
	setp.lt.f32 	%p32, %f191, 0f00000000;
	mov.f32 	%f214, 0f40490FDB;
	sub.f32 	%f215, %f214, %f213;
	selp.f32 	%f216, %f215, %f213, %p32;
	max.f32 	%f253, %f253, %f216;

$L__BB0_28:
	add.s32 	%r27, %r3, 1;
	@%p28 bra 	$L__BB0_30;
	bra.uni 	$L__BB0_29;

$L__BB0_30:
	add.s32 	%r96, %r33, -1;
	min.s32 	%r110, %r27, %r96;
	bra.uni 	$L__BB0_31;

$L__BB0_29:
	rem.s32 	%r94, %r27, %r33;
	add.s32 	%r95, %r94, %r33;
	rem.s32 	%r110, %r95, %r33;

$L__BB0_31:
	mad.lo.s32 	%r97, %r110, %r32, %r2;
	mad.lo.s32 	%r98, %r97, %r31, %r1;
	cvt.s64.s32 	%rd58, %r98;
	mul.wide.s32 	%rd59, %r98, 4;
	add.s64 	%rd60, %rd5, %rd59;
	add.s64 	%rd61, %rd4, %rd59;
	add.s64 	%rd62, %rd3, %rd59;
	ld.global.nc.f32 	%f217, [%rd62];
	ld.global.nc.f32 	%f218, [%rd60];
	ld.global.nc.f32 	%f219, [%rd61];
	mul.f32 	%f220, %f219, %f219;
	fma.rn.f32 	%f221, %f218, %f218, %f220;
	fma.rn.f32 	%f222, %f217, %f217, %f221;
	setp.eq.f32 	%p34, %f222, 0f00000000;
	selp.f32 	%f34, %f3, %f217, %p34;
	selp.f32 	%f33, %f2, %f219, %p34;
	selp.f32 	%f32, %f1, %f218, %p34;
	add.s64 	%rd63, %rd2, %rd58;
	ld.global.nc.u8 	%rs31, [%rd63];
	min.u16 	%rs34, %rs31, %rs1;
	cvt.u32.u16 	%r99, %rs34;
	max.u16 	%rs35, %rs31, %rs1;
	cvt.u32.u16 	%r100, %rs35;
	add.s32 	%r101, %r100, 1;
	mul.lo.s32 	%r102, %r101, %r100;
	shr.u32 	%r103, %r102, 1;
	add.s32 	%r104, %r103, %r99;
	mul.wide.s32 	%rd64, %r104, 4;
	add.s64 	%rd65, %rd1, %rd64;
	ld.global.nc.f32 	%f223, [%rd65];
	setp.eq.f32 	%p35, %f223, 0f00000000;
	@%p35 bra 	$L__BB0_33;

	mul.f32 	%f224, %f2, %f33;
	fma.rn.f32 	%f225, %f1, %f32, %f224;
	fma.rn.f32 	%f226, %f3, %f34, %f225;
	abs.f32 	%f227, %f226;
	mov.f32 	%f228, 0f3F800000;
	sub.f32 	%f229, %f228, %f227;
	mul.f32 	%f230, %f229, 0f3F000000;
	sqrt.rn.f32 	%f231, %f230;
	setp.gt.f32 	%p36, %f227, 0f3F11EB85;
	selp.f32 	%f232, %f231, %f227, %p36;
	mul.f32 	%f233, %f232, %f232;
	mov.f32 	%f234, 0f3C94D2E9;
	mov.f32 	%f235, 0f3D53F941;
	fma.rn.f32 	%f236, %f235, %f233, %f234;
	mov.f32 	%f237, 0f3D3F841F;
	fma.rn.f32 	%f238, %f236, %f233, %f237;
	mov.f32 	%f239, 0f3D994929;
	fma.rn.f32 	%f240, %f238, %f233, %f239;
	mov.f32 	%f241, 0f3E2AAB94;
	fma.rn.f32 	%f242, %f240, %f233, %f241;
	mul.f32 	%f243, %f233, %f242;
	fma.rn.f32 	%f244, %f243, %f232, %f232;
	add.f32 	%f245, %f244, %f244;
	mov.f32 	%f246, 0f3FC90FDB;
	sub.f32 	%f247, %f246, %f244;
	selp.f32 	%f248, %f245, %f247, %p36;
	setp.lt.f32 	%p37, %f226, 0f00000000;
	mov.f32 	%f249, 0f40490FDB;
	sub.f32 	%f250, %f249, %f248;
	selp.f32 	%f251, %f250, %f248, %p37;
	max.f32 	%f253, %f253, %f251;

$L__BB0_33:
	cvta.to.global.u64 	%rd66, %rd6;
	add.s64 	%rd68, %rd66, %rd12;
	st.global.f32 	[%rd68], %f253;

$L__BB0_34:
	ret;

}

`
)
