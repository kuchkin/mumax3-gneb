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

// CUDA handle for adddmibulk kernel
var adddmibulk_code cu.Function

// Stores the arguments for adddmibulk kernel invocation
type adddmibulk_args_t struct {
	arg_Hx      unsafe.Pointer
	arg_Hy      unsafe.Pointer
	arg_Hz      unsafe.Pointer
	arg_mx      unsafe.Pointer
	arg_my      unsafe.Pointer
	arg_mz      unsafe.Pointer
	arg_Ms_     unsafe.Pointer
	arg_Ms_mul  float32
	arg_aLUT2d  unsafe.Pointer
	arg_DLUT2d  unsafe.Pointer
	arg_regions unsafe.Pointer
	arg_cx      float32
	arg_cy      float32
	arg_cz      float32
	arg_Nx      int
	arg_Ny      int
	arg_Nz      int
	arg_PBC     byte
	arg_OpenBC  byte
	argptr      [19]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for adddmibulk kernel invocation
var adddmibulk_args adddmibulk_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	adddmibulk_args.argptr[0] = unsafe.Pointer(&adddmibulk_args.arg_Hx)
	adddmibulk_args.argptr[1] = unsafe.Pointer(&adddmibulk_args.arg_Hy)
	adddmibulk_args.argptr[2] = unsafe.Pointer(&adddmibulk_args.arg_Hz)
	adddmibulk_args.argptr[3] = unsafe.Pointer(&adddmibulk_args.arg_mx)
	adddmibulk_args.argptr[4] = unsafe.Pointer(&adddmibulk_args.arg_my)
	adddmibulk_args.argptr[5] = unsafe.Pointer(&adddmibulk_args.arg_mz)
	adddmibulk_args.argptr[6] = unsafe.Pointer(&adddmibulk_args.arg_Ms_)
	adddmibulk_args.argptr[7] = unsafe.Pointer(&adddmibulk_args.arg_Ms_mul)
	adddmibulk_args.argptr[8] = unsafe.Pointer(&adddmibulk_args.arg_aLUT2d)
	adddmibulk_args.argptr[9] = unsafe.Pointer(&adddmibulk_args.arg_DLUT2d)
	adddmibulk_args.argptr[10] = unsafe.Pointer(&adddmibulk_args.arg_regions)
	adddmibulk_args.argptr[11] = unsafe.Pointer(&adddmibulk_args.arg_cx)
	adddmibulk_args.argptr[12] = unsafe.Pointer(&adddmibulk_args.arg_cy)
	adddmibulk_args.argptr[13] = unsafe.Pointer(&adddmibulk_args.arg_cz)
	adddmibulk_args.argptr[14] = unsafe.Pointer(&adddmibulk_args.arg_Nx)
	adddmibulk_args.argptr[15] = unsafe.Pointer(&adddmibulk_args.arg_Ny)
	adddmibulk_args.argptr[16] = unsafe.Pointer(&adddmibulk_args.arg_Nz)
	adddmibulk_args.argptr[17] = unsafe.Pointer(&adddmibulk_args.arg_PBC)
	adddmibulk_args.argptr[18] = unsafe.Pointer(&adddmibulk_args.arg_OpenBC)
}

// Wrapper for adddmibulk CUDA kernel, asynchronous.
func k_adddmibulk_async(Hx unsafe.Pointer, Hy unsafe.Pointer, Hz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, Ms_ unsafe.Pointer, Ms_mul float32, aLUT2d unsafe.Pointer, DLUT2d unsafe.Pointer, regions unsafe.Pointer, cx float32, cy float32, cz float32, Nx int, Ny int, Nz int, PBC byte, OpenBC byte, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("adddmibulk")
	}

	adddmibulk_args.Lock()
	defer adddmibulk_args.Unlock()

	if adddmibulk_code == 0 {
		adddmibulk_code = fatbinLoad(adddmibulk_map, "adddmibulk")
	}

	adddmibulk_args.arg_Hx = Hx
	adddmibulk_args.arg_Hy = Hy
	adddmibulk_args.arg_Hz = Hz
	adddmibulk_args.arg_mx = mx
	adddmibulk_args.arg_my = my
	adddmibulk_args.arg_mz = mz
	adddmibulk_args.arg_Ms_ = Ms_
	adddmibulk_args.arg_Ms_mul = Ms_mul
	adddmibulk_args.arg_aLUT2d = aLUT2d
	adddmibulk_args.arg_DLUT2d = DLUT2d
	adddmibulk_args.arg_regions = regions
	adddmibulk_args.arg_cx = cx
	adddmibulk_args.arg_cy = cy
	adddmibulk_args.arg_cz = cz
	adddmibulk_args.arg_Nx = Nx
	adddmibulk_args.arg_Ny = Ny
	adddmibulk_args.arg_Nz = Nz
	adddmibulk_args.arg_PBC = PBC
	adddmibulk_args.arg_OpenBC = OpenBC

	args := adddmibulk_args.argptr[:]
	cu.LaunchKernel(adddmibulk_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("adddmibulk")
	}
}

// maps compute capability on PTX code for adddmibulk kernel.
var adddmibulk_map = map[int]string{0: "",
	50: adddmibulk_ptx_50}

// adddmibulk PTX code for various compute capabilities.
const (
	adddmibulk_ptx_50 = `
.version 7.5
.target sm_50
.address_size 64

	// .globl	adddmibulk

.visible .entry adddmibulk(
	.param .u64 adddmibulk_param_0,
	.param .u64 adddmibulk_param_1,
	.param .u64 adddmibulk_param_2,
	.param .u64 adddmibulk_param_3,
	.param .u64 adddmibulk_param_4,
	.param .u64 adddmibulk_param_5,
	.param .u64 adddmibulk_param_6,
	.param .f32 adddmibulk_param_7,
	.param .u64 adddmibulk_param_8,
	.param .u64 adddmibulk_param_9,
	.param .u64 adddmibulk_param_10,
	.param .f32 adddmibulk_param_11,
	.param .f32 adddmibulk_param_12,
	.param .f32 adddmibulk_param_13,
	.param .u32 adddmibulk_param_14,
	.param .u32 adddmibulk_param_15,
	.param .u32 adddmibulk_param_16,
	.param .u8 adddmibulk_param_17,
	.param .u8 adddmibulk_param_18
)
{
	.reg .pred 	%p<64>;
	.reg .b16 	%rs<49>;
	.reg .f32 	%f<363>;
	.reg .b32 	%r<111>;
	.reg .b64 	%rd<87>;


	ld.param.u8 	%rs18, [adddmibulk_param_18];
	ld.param.u8 	%rs17, [adddmibulk_param_17];
	ld.param.u64 	%rd7, [adddmibulk_param_0];
	ld.param.u64 	%rd8, [adddmibulk_param_1];
	ld.param.u64 	%rd9, [adddmibulk_param_2];
	ld.param.u64 	%rd11, [adddmibulk_param_3];
	ld.param.u64 	%rd12, [adddmibulk_param_4];
	ld.param.u64 	%rd13, [adddmibulk_param_5];
	ld.param.u64 	%rd10, [adddmibulk_param_6];
	ld.param.f32 	%f361, [adddmibulk_param_7];
	ld.param.u64 	%rd14, [adddmibulk_param_8];
	ld.param.u64 	%rd15, [adddmibulk_param_9];
	ld.param.u64 	%rd16, [adddmibulk_param_10];
	ld.param.f32 	%f180, [adddmibulk_param_11];
	ld.param.f32 	%f181, [adddmibulk_param_12];
	ld.param.f32 	%f182, [adddmibulk_param_13];
	ld.param.u32 	%r37, [adddmibulk_param_14];
	ld.param.u32 	%r38, [adddmibulk_param_15];
	ld.param.u32 	%r39, [adddmibulk_param_16];
	cvta.to.global.u64 	%rd1, %rd15;
	cvta.to.global.u64 	%rd2, %rd14;
	cvta.to.global.u64 	%rd3, %rd16;
	cvta.to.global.u64 	%rd4, %rd13;
	cvta.to.global.u64 	%rd5, %rd12;
	cvta.to.global.u64 	%rd6, %rd11;
	mov.u32 	%r40, %ntid.x;
	mov.u32 	%r41, %ctaid.x;
	mov.u32 	%r42, %tid.x;
	mad.lo.s32 	%r1, %r41, %r40, %r42;
	mov.u32 	%r43, %ntid.y;
	mov.u32 	%r44, %ctaid.y;
	mov.u32 	%r45, %tid.y;
	mad.lo.s32 	%r2, %r44, %r43, %r45;
	mov.u32 	%r46, %ntid.z;
	mov.u32 	%r47, %ctaid.z;
	mov.u32 	%r48, %tid.z;
	mad.lo.s32 	%r3, %r47, %r46, %r48;
	setp.ge.s32 	%p1, %r1, %r37;
	setp.ge.s32 	%p2, %r2, %r38;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r39;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_74;

	mul.lo.s32 	%r4, %r3, %r38;
	add.s32 	%r49, %r4, %r2;
	mul.lo.s32 	%r5, %r49, %r37;
	add.s32 	%r6, %r5, %r1;
	mul.wide.s32 	%rd17, %r6, 4;
	add.s64 	%rd18, %rd6, %rd17;
	cvt.s64.s32 	%rd19, %r6;
	add.s64 	%rd20, %rd5, %rd17;
	add.s64 	%rd21, %rd4, %rd17;
	add.s64 	%rd22, %rd3, %rd19;
	ld.global.nc.u8 	%rs1, [%rd22];
	ld.global.nc.f32 	%f1, [%rd18];
	ld.global.nc.f32 	%f5, [%rd20];
	mul.f32 	%f185, %f5, %f5;
	fma.rn.f32 	%f186, %f1, %f1, %f185;
	ld.global.nc.f32 	%f6, [%rd21];
	fma.rn.f32 	%f187, %f6, %f6, %f186;
	setp.eq.f32 	%p6, %f187, 0f00000000;
	@%p6 bra 	$L__BB0_74;

	and.b16  	%rs2, %rs17, 1;
	setp.eq.s16 	%p7, %rs2, 0;
	add.s32 	%r7, %r1, -1;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	max.s32 	%r105, %r7, 0;
	bra.uni 	$L__BB0_5;

$L__BB0_3:
	rem.s32 	%r50, %r7, %r37;
	add.s32 	%r51, %r50, %r37;
	rem.s32 	%r105, %r51, %r37;

$L__BB0_5:
	add.s32 	%r11, %r105, %r5;
	setp.lt.s32 	%p9, %r1, 1;
	mov.f32 	%f312, 0f00000000;
	and.pred  	%p10, %p9, %p7;
	mov.f32 	%f311, %f312;
	mov.f32 	%f310, %f312;
	@%p10 bra 	$L__BB0_7;

	mul.wide.s32 	%rd23, %r11, 4;
	add.s64 	%rd24, %rd6, %rd23;
	ld.global.nc.f32 	%f310, [%rd24];
	add.s64 	%rd25, %rd5, %rd23;
	ld.global.nc.f32 	%f311, [%rd25];
	add.s64 	%rd26, %rd4, %rd23;
	ld.global.nc.f32 	%f312, [%rd26];

$L__BB0_7:
	mul.f32 	%f191, %f310, %f310;
	fma.rn.f32 	%f192, %f311, %f311, %f191;
	fma.rn.f32 	%f13, %f312, %f312, %f192;
	setp.eq.f32 	%p11, %f13, 0f00000000;
	mov.u16 	%rs43, %rs1;
	@%p11 bra 	$L__BB0_9;

	cvt.s64.s32 	%rd27, %r11;
	add.s64 	%rd28, %rd3, %rd27;
	ld.global.nc.u8 	%rs43, [%rd28];

$L__BB0_9:
	min.u16 	%rs21, %rs43, %rs1;
	cvt.u32.u16 	%r52, %rs21;
	max.u16 	%rs22, %rs43, %rs1;
	cvt.u32.u16 	%r53, %rs22;
	add.s32 	%r54, %r53, 1;
	mul.lo.s32 	%r55, %r54, %r53;
	shr.u32 	%r56, %r55, 1;
	add.s32 	%r57, %r56, %r52;
	mul.wide.s32 	%rd29, %r57, 4;
	add.s64 	%rd30, %rd2, %rd29;
	add.s64 	%rd31, %rd1, %rd29;
	ld.global.nc.f32 	%f196, [%rd30];
	add.f32 	%f14, %f196, %f196;
	ld.global.nc.f32 	%f15, [%rd31];
	setp.ne.s16 	%p12, %rs18, 0;
	mov.f32 	%f322, 0f00000000;
	and.pred  	%p14, %p12, %p11;
	mov.f32 	%f323, %f322;
	mov.f32 	%f324, %f322;
	@%p14 bra 	$L__BB0_13;

	setp.neu.f32 	%p15, %f13, 0f00000000;
	@%p15 bra 	$L__BB0_12;

	div.rn.f32 	%f197, %f15, %f14;
	mul.f32 	%f198, %f197, %f180;
	fma.rn.f32 	%f311, %f6, %f198, %f5;
	mul.f32 	%f199, %f5, %f198;
	sub.f32 	%f312, %f6, %f199;
	mov.f32 	%f310, %f1;

$L__BB0_12:
	mul.f32 	%f200, %f180, %f180;
	div.rn.f32 	%f201, %f14, %f200;
	sub.f32 	%f202, %f310, %f1;
	sub.f32 	%f203, %f311, %f5;
	sub.f32 	%f204, %f312, %f6;
	fma.rn.f32 	%f322, %f201, %f202, 0f00000000;
	fma.rn.f32 	%f205, %f201, %f203, 0f00000000;
	fma.rn.f32 	%f206, %f201, %f204, 0f00000000;
	div.rn.f32 	%f207, %f15, %f180;
	mul.f32 	%f208, %f207, %f312;
	sub.f32 	%f323, %f205, %f208;
	fma.rn.f32 	%f324, %f207, %f311, %f206;

$L__BB0_13:
	add.s32 	%r12, %r1, 1;
	@%p7 bra 	$L__BB0_15;
	bra.uni 	$L__BB0_14;

$L__BB0_15:
	add.s32 	%r60, %r37, -1;
	min.s32 	%r106, %r12, %r60;
	bra.uni 	$L__BB0_16;

$L__BB0_14:
	rem.s32 	%r58, %r12, %r37;
	add.s32 	%r59, %r58, %r37;
	rem.s32 	%r106, %r59, %r37;

$L__BB0_16:
	add.s32 	%r16, %r106, %r5;
	setp.ge.s32 	%p17, %r12, %r37;
	mov.f32 	%f321, 0f00000000;
	and.pred  	%p19, %p17, %p7;
	mov.f32 	%f320, %f321;
	mov.f32 	%f319, %f321;
	@%p19 bra 	$L__BB0_18;

	mul.wide.s32 	%rd32, %r16, 4;
	add.s64 	%rd33, %rd6, %rd32;
	ld.global.nc.f32 	%f319, [%rd33];
	add.s64 	%rd34, %rd5, %rd32;
	ld.global.nc.f32 	%f320, [%rd34];
	add.s64 	%rd35, %rd4, %rd32;
	ld.global.nc.f32 	%f321, [%rd35];

$L__BB0_18:
	mul.f32 	%f212, %f319, %f319;
	fma.rn.f32 	%f213, %f320, %f320, %f212;
	fma.rn.f32 	%f41, %f321, %f321, %f213;
	setp.eq.f32 	%p20, %f41, 0f00000000;
	mov.u16 	%rs44, %rs1;
	@%p20 bra 	$L__BB0_20;

	cvt.s64.s32 	%rd36, %r16;
	add.s64 	%rd37, %rd3, %rd36;
	ld.global.nc.u8 	%rs44, [%rd37];

$L__BB0_20:
	min.u16 	%rs25, %rs44, %rs1;
	cvt.u32.u16 	%r61, %rs25;
	max.u16 	%rs26, %rs44, %rs1;
	cvt.u32.u16 	%r62, %rs26;
	add.s32 	%r63, %r62, 1;
	mul.lo.s32 	%r64, %r63, %r62;
	shr.u32 	%r65, %r64, 1;
	add.s32 	%r66, %r65, %r61;
	mul.wide.s32 	%rd38, %r66, 4;
	add.s64 	%rd39, %rd2, %rd38;
	add.s64 	%rd40, %rd1, %rd38;
	ld.global.nc.f32 	%f214, [%rd39];
	add.f32 	%f42, %f214, %f214;
	ld.global.nc.f32 	%f43, [%rd40];
	and.pred  	%p23, %p12, %p20;
	@%p23 bra 	$L__BB0_24;

	setp.neu.f32 	%p24, %f41, 0f00000000;
	@%p24 bra 	$L__BB0_23;

	div.rn.f32 	%f215, %f43, %f42;
	mul.f32 	%f216, %f215, %f180;
	mul.f32 	%f217, %f6, %f216;
	sub.f32 	%f320, %f5, %f217;
	fma.rn.f32 	%f321, %f5, %f216, %f6;
	mov.f32 	%f319, %f1;

$L__BB0_23:
	mul.f32 	%f218, %f180, %f180;
	div.rn.f32 	%f219, %f42, %f218;
	sub.f32 	%f220, %f319, %f1;
	sub.f32 	%f221, %f320, %f5;
	sub.f32 	%f222, %f321, %f6;
	fma.rn.f32 	%f322, %f219, %f220, %f322;
	fma.rn.f32 	%f223, %f219, %f221, %f323;
	fma.rn.f32 	%f224, %f219, %f222, %f324;
	div.rn.f32 	%f225, %f43, %f180;
	fma.rn.f32 	%f323, %f225, %f321, %f223;
	mul.f32 	%f226, %f225, %f320;
	sub.f32 	%f324, %f224, %f226;

$L__BB0_24:
	and.b16  	%rs7, %rs17, 2;
	setp.eq.s16 	%p25, %rs7, 0;
	add.s32 	%r17, %r2, -1;
	@%p25 bra 	$L__BB0_26;
	bra.uni 	$L__BB0_25;

$L__BB0_26:
	max.s32 	%r107, %r17, 0;
	bra.uni 	$L__BB0_27;

$L__BB0_25:
	rem.s32 	%r67, %r17, %r38;
	add.s32 	%r68, %r67, %r38;
	rem.s32 	%r107, %r68, %r38;

$L__BB0_27:
	add.s32 	%r69, %r107, %r4;
	mad.lo.s32 	%r21, %r69, %r37, %r1;
	setp.lt.s32 	%p27, %r2, 1;
	mov.f32 	%f330, 0f00000000;
	and.pred  	%p28, %p27, %p25;
	mov.f32 	%f329, %f330;
	mov.f32 	%f328, %f330;
	@%p28 bra 	$L__BB0_29;

	mul.wide.s32 	%rd41, %r21, 4;
	add.s64 	%rd42, %rd6, %rd41;
	ld.global.nc.f32 	%f328, [%rd42];
	add.s64 	%rd43, %rd5, %rd41;
	ld.global.nc.f32 	%f329, [%rd43];
	add.s64 	%rd44, %rd4, %rd41;
	ld.global.nc.f32 	%f330, [%rd44];

$L__BB0_29:
	mul.f32 	%f230, %f328, %f328;
	fma.rn.f32 	%f231, %f329, %f329, %f230;
	fma.rn.f32 	%f69, %f330, %f330, %f231;
	setp.eq.f32 	%p29, %f69, 0f00000000;
	mov.u16 	%rs45, %rs1;
	@%p29 bra 	$L__BB0_31;

	cvt.s64.s32 	%rd45, %r21;
	add.s64 	%rd46, %rd3, %rd45;
	ld.global.nc.u8 	%rs45, [%rd46];

$L__BB0_31:
	min.u16 	%rs29, %rs45, %rs1;
	cvt.u32.u16 	%r70, %rs29;
	max.u16 	%rs30, %rs45, %rs1;
	cvt.u32.u16 	%r71, %rs30;
	add.s32 	%r72, %r71, 1;
	mul.lo.s32 	%r73, %r72, %r71;
	shr.u32 	%r74, %r73, 1;
	add.s32 	%r75, %r74, %r70;
	mul.wide.s32 	%rd47, %r75, 4;
	add.s64 	%rd48, %rd2, %rd47;
	add.s64 	%rd49, %rd1, %rd47;
	ld.global.nc.f32 	%f232, [%rd48];
	add.f32 	%f70, %f232, %f232;
	ld.global.nc.f32 	%f71, [%rd49];
	and.pred  	%p32, %p12, %p29;
	@%p32 bra 	$L__BB0_35;

	setp.neu.f32 	%p33, %f69, 0f00000000;
	@%p33 bra 	$L__BB0_34;

	div.rn.f32 	%f233, %f71, %f70;
	mul.f32 	%f234, %f233, %f181;
	mul.f32 	%f235, %f6, %f234;
	sub.f32 	%f328, %f1, %f235;
	fma.rn.f32 	%f330, %f1, %f234, %f6;
	mov.f32 	%f329, %f5;

$L__BB0_34:
	mul.f32 	%f236, %f181, %f181;
	div.rn.f32 	%f237, %f70, %f236;
	sub.f32 	%f238, %f328, %f1;
	sub.f32 	%f239, %f329, %f5;
	sub.f32 	%f240, %f330, %f6;
	fma.rn.f32 	%f241, %f237, %f238, %f322;
	fma.rn.f32 	%f323, %f237, %f239, %f323;
	fma.rn.f32 	%f242, %f237, %f240, %f324;
	div.rn.f32 	%f243, %f71, %f181;
	fma.rn.f32 	%f322, %f243, %f330, %f241;
	mul.f32 	%f244, %f243, %f328;
	sub.f32 	%f324, %f242, %f244;

$L__BB0_35:
	add.s32 	%r22, %r2, 1;
	@%p25 bra 	$L__BB0_37;
	bra.uni 	$L__BB0_36;

$L__BB0_37:
	add.s32 	%r78, %r38, -1;
	min.s32 	%r108, %r22, %r78;
	bra.uni 	$L__BB0_38;

$L__BB0_36:
	rem.s32 	%r76, %r22, %r38;
	add.s32 	%r77, %r76, %r38;
	rem.s32 	%r108, %r77, %r38;

$L__BB0_38:
	add.s32 	%r79, %r108, %r4;
	mad.lo.s32 	%r26, %r79, %r37, %r1;
	setp.ge.s32 	%p35, %r22, %r38;
	mov.f32 	%f339, 0f00000000;
	and.pred  	%p37, %p35, %p25;
	mov.f32 	%f338, %f339;
	mov.f32 	%f337, %f339;
	@%p37 bra 	$L__BB0_40;

	mul.wide.s32 	%rd50, %r26, 4;
	add.s64 	%rd51, %rd6, %rd50;
	ld.global.nc.f32 	%f337, [%rd51];
	add.s64 	%rd52, %rd5, %rd50;
	ld.global.nc.f32 	%f338, [%rd52];
	add.s64 	%rd53, %rd4, %rd50;
	ld.global.nc.f32 	%f339, [%rd53];

$L__BB0_40:
	mul.f32 	%f248, %f337, %f337;
	fma.rn.f32 	%f249, %f338, %f338, %f248;
	fma.rn.f32 	%f97, %f339, %f339, %f249;
	setp.eq.f32 	%p38, %f97, 0f00000000;
	mov.u16 	%rs46, %rs1;
	@%p38 bra 	$L__BB0_42;

	cvt.s64.s32 	%rd54, %r26;
	add.s64 	%rd55, %rd3, %rd54;
	ld.global.nc.u8 	%rs46, [%rd55];

$L__BB0_42:
	min.u16 	%rs33, %rs46, %rs1;
	cvt.u32.u16 	%r80, %rs33;
	max.u16 	%rs34, %rs46, %rs1;
	cvt.u32.u16 	%r81, %rs34;
	add.s32 	%r82, %r81, 1;
	mul.lo.s32 	%r83, %r82, %r81;
	shr.u32 	%r84, %r83, 1;
	add.s32 	%r85, %r84, %r80;
	mul.wide.s32 	%rd56, %r85, 4;
	add.s64 	%rd57, %rd2, %rd56;
	add.s64 	%rd58, %rd1, %rd56;
	ld.global.nc.f32 	%f250, [%rd57];
	add.f32 	%f98, %f250, %f250;
	ld.global.nc.f32 	%f99, [%rd58];
	and.pred  	%p41, %p12, %p38;
	@%p41 bra 	$L__BB0_46;

	setp.neu.f32 	%p42, %f97, 0f00000000;
	@%p42 bra 	$L__BB0_45;

	div.rn.f32 	%f251, %f99, %f98;
	mul.f32 	%f252, %f251, %f181;
	fma.rn.f32 	%f337, %f6, %f252, %f1;
	mul.f32 	%f253, %f1, %f252;
	sub.f32 	%f339, %f6, %f253;
	mov.f32 	%f338, %f5;

$L__BB0_45:
	mul.f32 	%f254, %f181, %f181;
	div.rn.f32 	%f255, %f98, %f254;
	sub.f32 	%f256, %f337, %f1;
	sub.f32 	%f257, %f338, %f5;
	sub.f32 	%f258, %f339, %f6;
	fma.rn.f32 	%f259, %f255, %f256, %f322;
	fma.rn.f32 	%f323, %f255, %f257, %f323;
	fma.rn.f32 	%f260, %f255, %f258, %f324;
	div.rn.f32 	%f261, %f99, %f181;
	mul.f32 	%f262, %f261, %f339;
	sub.f32 	%f322, %f259, %f262;
	fma.rn.f32 	%f324, %f261, %f337, %f260;

$L__BB0_46:
	setp.eq.s32 	%p43, %r39, 1;
	@%p43 bra 	$L__BB0_69;

	and.b16  	%rs12, %rs17, 4;
	setp.eq.s16 	%p44, %rs12, 0;
	add.s32 	%r27, %r3, -1;
	@%p44 bra 	$L__BB0_49;
	bra.uni 	$L__BB0_48;

$L__BB0_49:
	max.s32 	%r109, %r27, 0;
	bra.uni 	$L__BB0_50;

$L__BB0_48:
	rem.s32 	%r86, %r27, %r39;
	add.s32 	%r87, %r86, %r39;
	rem.s32 	%r109, %r87, %r39;

$L__BB0_50:
	mad.lo.s32 	%r88, %r109, %r38, %r2;
	mad.lo.s32 	%r31, %r88, %r37, %r1;
	setp.lt.s32 	%p46, %r3, 1;
	mov.f32 	%f348, 0f00000000;
	and.pred  	%p47, %p46, %p44;
	mov.f32 	%f347, %f348;
	mov.f32 	%f346, %f348;
	@%p47 bra 	$L__BB0_52;

	mul.wide.s32 	%rd59, %r31, 4;
	add.s64 	%rd60, %rd6, %rd59;
	ld.global.nc.f32 	%f346, [%rd60];
	add.s64 	%rd61, %rd5, %rd59;
	ld.global.nc.f32 	%f347, [%rd61];
	add.s64 	%rd62, %rd4, %rd59;
	ld.global.nc.f32 	%f348, [%rd62];

$L__BB0_52:
	mul.f32 	%f266, %f346, %f346;
	fma.rn.f32 	%f267, %f347, %f347, %f266;
	fma.rn.f32 	%f125, %f348, %f348, %f267;
	setp.eq.f32 	%p48, %f125, 0f00000000;
	mov.u16 	%rs47, %rs1;
	@%p48 bra 	$L__BB0_54;

	cvt.s64.s32 	%rd63, %r31;
	add.s64 	%rd64, %rd3, %rd63;
	ld.global.nc.u8 	%rs47, [%rd64];

$L__BB0_54:
	min.u16 	%rs37, %rs47, %rs1;
	cvt.u32.u16 	%r89, %rs37;
	max.u16 	%rs38, %rs47, %rs1;
	cvt.u32.u16 	%r90, %rs38;
	add.s32 	%r91, %r90, 1;
	mul.lo.s32 	%r92, %r91, %r90;
	shr.u32 	%r93, %r92, 1;
	add.s32 	%r94, %r93, %r89;
	mul.wide.s32 	%rd65, %r94, 4;
	add.s64 	%rd66, %rd2, %rd65;
	add.s64 	%rd67, %rd1, %rd65;
	ld.global.nc.f32 	%f268, [%rd66];
	add.f32 	%f126, %f268, %f268;
	ld.global.nc.f32 	%f127, [%rd67];
	and.pred  	%p51, %p12, %p48;
	@%p51 bra 	$L__BB0_58;

	setp.neu.f32 	%p52, %f125, 0f00000000;
	@%p52 bra 	$L__BB0_57;

	div.rn.f32 	%f269, %f127, %f126;
	mul.f32 	%f270, %f269, %f182;
	fma.rn.f32 	%f346, %f5, %f270, %f1;
	mul.f32 	%f271, %f1, %f270;
	sub.f32 	%f347, %f5, %f271;
	mov.f32 	%f348, %f6;

$L__BB0_57:
	mul.f32 	%f272, %f182, %f182;
	div.rn.f32 	%f273, %f126, %f272;
	sub.f32 	%f274, %f346, %f1;
	sub.f32 	%f275, %f347, %f5;
	sub.f32 	%f276, %f348, %f6;
	fma.rn.f32 	%f277, %f273, %f274, %f322;
	fma.rn.f32 	%f278, %f273, %f275, %f323;
	fma.rn.f32 	%f324, %f273, %f276, %f324;
	div.rn.f32 	%f279, %f127, %f182;
	mul.f32 	%f280, %f279, %f347;
	sub.f32 	%f322, %f277, %f280;
	fma.rn.f32 	%f323, %f279, %f346, %f278;

$L__BB0_58:
	add.s32 	%r32, %r3, 1;
	@%p44 bra 	$L__BB0_60;
	bra.uni 	$L__BB0_59;

$L__BB0_60:
	add.s32 	%r97, %r39, -1;
	min.s32 	%r110, %r32, %r97;
	bra.uni 	$L__BB0_61;

$L__BB0_59:
	rem.s32 	%r95, %r32, %r39;
	add.s32 	%r96, %r95, %r39;
	rem.s32 	%r110, %r96, %r39;

$L__BB0_61:
	mad.lo.s32 	%r98, %r110, %r38, %r2;
	mad.lo.s32 	%r36, %r98, %r37, %r1;
	setp.ge.s32 	%p54, %r32, %r39;
	mov.f32 	%f357, 0f00000000;
	and.pred  	%p56, %p54, %p44;
	mov.f32 	%f356, %f357;
	mov.f32 	%f355, %f357;
	@%p56 bra 	$L__BB0_63;

	mul.wide.s32 	%rd68, %r36, 4;
	add.s64 	%rd69, %rd6, %rd68;
	ld.global.nc.f32 	%f355, [%rd69];
	add.s64 	%rd70, %rd5, %rd68;
	ld.global.nc.f32 	%f356, [%rd70];
	add.s64 	%rd71, %rd4, %rd68;
	ld.global.nc.f32 	%f357, [%rd71];

$L__BB0_63:
	mul.f32 	%f284, %f355, %f355;
	fma.rn.f32 	%f285, %f356, %f356, %f284;
	fma.rn.f32 	%f153, %f357, %f357, %f285;
	setp.eq.f32 	%p57, %f153, 0f00000000;
	mov.u16 	%rs48, %rs1;
	@%p57 bra 	$L__BB0_65;

	cvt.s64.s32 	%rd72, %r36;
	add.s64 	%rd73, %rd3, %rd72;
	ld.global.nc.u8 	%rs48, [%rd73];

$L__BB0_65:
	min.u16 	%rs41, %rs48, %rs1;
	cvt.u32.u16 	%r99, %rs41;
	max.u16 	%rs42, %rs48, %rs1;
	cvt.u32.u16 	%r100, %rs42;
	add.s32 	%r101, %r100, 1;
	mul.lo.s32 	%r102, %r101, %r100;
	shr.u32 	%r103, %r102, 1;
	add.s32 	%r104, %r103, %r99;
	mul.wide.s32 	%rd74, %r104, 4;
	add.s64 	%rd75, %rd2, %rd74;
	add.s64 	%rd76, %rd1, %rd74;
	ld.global.nc.f32 	%f286, [%rd75];
	add.f32 	%f154, %f286, %f286;
	ld.global.nc.f32 	%f155, [%rd76];
	and.pred  	%p60, %p12, %p57;
	@%p60 bra 	$L__BB0_69;

	setp.neu.f32 	%p61, %f153, 0f00000000;
	@%p61 bra 	$L__BB0_68;

	div.rn.f32 	%f287, %f155, %f154;
	mul.f32 	%f288, %f287, %f182;
	mul.f32 	%f289, %f5, %f288;
	sub.f32 	%f355, %f1, %f289;
	fma.rn.f32 	%f356, %f1, %f288, %f5;
	mov.f32 	%f357, %f6;

$L__BB0_68:
	mul.f32 	%f290, %f182, %f182;
	div.rn.f32 	%f291, %f154, %f290;
	sub.f32 	%f292, %f355, %f1;
	sub.f32 	%f293, %f356, %f5;
	sub.f32 	%f294, %f357, %f6;
	fma.rn.f32 	%f295, %f291, %f292, %f322;
	fma.rn.f32 	%f296, %f291, %f293, %f323;
	fma.rn.f32 	%f324, %f291, %f294, %f324;
	div.rn.f32 	%f297, %f155, %f182;
	fma.rn.f32 	%f322, %f297, %f356, %f295;
	mul.f32 	%f298, %f297, %f355;
	sub.f32 	%f323, %f296, %f298;

$L__BB0_69:
	setp.eq.s64 	%p62, %rd10, 0;
	@%p62 bra 	$L__BB0_71;

	cvta.to.global.u64 	%rd77, %rd10;
	add.s64 	%rd79, %rd77, %rd17;
	ld.global.nc.f32 	%f299, [%rd79];
	mul.f32 	%f361, %f299, %f361;

$L__BB0_71:
	setp.eq.f32 	%p63, %f361, 0f00000000;
	mov.f32 	%f362, 0f00000000;
	@%p63 bra 	$L__BB0_73;

	rcp.rn.f32 	%f362, %f361;

$L__BB0_73:
	cvta.to.global.u64 	%rd80, %rd7;
	add.s64 	%rd82, %rd80, %rd17;
	ld.global.f32 	%f301, [%rd82];
	fma.rn.f32 	%f302, %f322, %f362, %f301;
	st.global.f32 	[%rd82], %f302;
	cvta.to.global.u64 	%rd83, %rd8;
	add.s64 	%rd84, %rd83, %rd17;
	ld.global.f32 	%f303, [%rd84];
	fma.rn.f32 	%f304, %f323, %f362, %f303;
	st.global.f32 	[%rd84], %f304;
	cvta.to.global.u64 	%rd85, %rd9;
	add.s64 	%rd86, %rd85, %rd17;
	ld.global.f32 	%f305, [%rd86];
	fma.rn.f32 	%f306, %f324, %f362, %f305;
	st.global.f32 	[%rd86], %f306;

$L__BB0_74:
	ret;

}

`
)
