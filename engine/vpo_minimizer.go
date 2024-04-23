package engine

//char outputfilename[200] = "table.csv";
// Minimize follows the steepest descent method as per Exl et al., JAP 115, 17D118 (2014).

import (
	"github.com/kuchkin/mumax3-gneb/cuda"
	"github.com/kuchkin/mumax3-gneb/data"
	"math"
	"unsafe"
)

/*
#include<stdio.h>
char shortBufer2[100];
void my_print2(int noi, float *coord,float *dist,float *energy,float *maxtorq, char *outputfilename) {
	FILE * pFile;
	pFile = fopen (outputfilename,"w");
    if (pFile!=NULL){
    }
    float prev = 0.0;
    float distance = 0.0;
    for(int i = 0;i<noi; i++){
    	if(i==(noi-1)){
    		coord[i] = coord[i-1];
    		dist[i] = dist[i-1];
    	}
    	prev += coord[i];
    	distance += dist[i];
    	snprintf(shortBufer2,100,"%i, %0.15f, %0.15f, %0.15f\n",i,energy[i],maxtorq[i],prev);
		fputs (shortBufer2,pFile);
    }
    fclose (pFile);
}
void my_print_interp(int noi, float *coord,float *tang,float *dist,float *energy, char *outputfilename,float pref) {
	FILE * pFile;
	pFile = fopen (outputfilename,"w");
    if (pFile!=NULL){
    }
    float distance = 0.0, prev = 0.0;
    int Points = 10;
    for(int i = 0;i<(noi-1); i++){


    	float x1 = 1.0*i, x2 = 1.0*(i+1);
    	float y1 = energy[i], y2 = energy[i+1];
    	float d1 = -1.0*pref*dist[i]/tang[i], d2 = -1.0*pref*dist[i+1]/tang[i+1];
    	if(i==0) d1 = 0.0;
    	if(i == (noi-2)) d2 = 0.0;
    	float a0 = x2*(x1*(x2-x1)*(d2*x1+d1*x2)-x2*(x2-3.*x1)*y1) + x1*x1*(x1-3.*x2)*y2;
    	float a1 = d2*x1*(x1-x2)*(x1+2.*x2) - x2*(d1*(x2*x2+x1*x2-2.*x1*x1) + 6.*x1*(y1-y2));
    	float a2 = 3.*(x1+x2)*(y1-y2) + d2*(x2*x2+x1*x2-2.*x1*x1) - d1*(x1-x2)*(x1+2.*x2);
    	float a3 = (d1+d2)*(x1-x2)-2.*y1+2.*y2;
    	for(int n = 0; n<Points; n++){
    		float x = x1 + (1.0*n)/(1.0*Points);
    		float y = -1.0*(a3*x*x*x + a2*x*x + a1*x + a0);
    		snprintf(shortBufer2,100,"%f, %0.15f, %0.15f\n",x,y,prev + (1.0*n)*(coord[i])/(1.0*Points));
			fputs (shortBufer2,pFile);
    	}
    	prev += coord[i];
    }
    snprintf(shortBufer2,100,"%f, %0.15f, %0.15f\n",1.0*(noi-1),energy[noi-1],prev);
	fputs (shortBufer2,pFile);
    fclose (pFile);
}
void vpo_print(int noi, float velocity,float torq,float vf,float k0k0, char *outputfilename) {
	FILE * pFile;
	pFile = fopen (outputfilename,"a+");
    if (pFile!=NULL){
    }
    snprintf(shortBufer2,100,"%i, %0.15f, %0.15f, %0.15f, %0.15f\n",noi,velocity,torq,vf,k0k0);
	fputs (shortBufer2,pFile);
    fclose (pFile);
}
void test_print(float vv,float dat, char *outputfilename) {
	FILE * pFile;
	pFile = fopen (outputfilename,"a+");
    if (pFile!=NULL){
    }
    snprintf(shortBufer2,100,"%0.15f,%0.15f\n",vv,dat);
	fputs (shortBufer2,pFile);
    fclose (pFile);
}
*/
import "C"

var (
	mass              = 1.0
	MaxForce          = 100.0
	CIGNEB            = 0
	Saddle            = 0
	InterpolateEnergy = 0
	PPP               = 1.0
)

func init() {
	DeclFunc("VPOminimize", VPOminimize, "Use gradient descent method to zeros the forces (or energy gradients)")
	DeclVar("mass", &mass, "mass")
	DeclVar("MaxForce", &MaxForce, "MaxForce")
	DeclVar("CIGNEB", &CIGNEB, "CIGNEB")
	DeclVar("Saddle", &Saddle, "Saddle")
	DeclVar("InterpolateEnergy", &InterpolateEnergy, "InterpolateEnergy")
	DeclVar("PPP", &PPP, "PPP")
}

type VPOminimizer struct {
	k   *data.Slice // torque saved to calculate time step
	vel *data.Slice // torque saved to calculate time step

}

func (mini *VPOminimizer) Step() {
	m := M.Buffer()
	size := m.Size()

	// vel := cuda.Buffer(3, size)
	// defer cuda.Recycle(vel)

	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)

	if mini.k == nil {
		mini.k = cuda.Buffer(3, size)
		// torqueFn(mini.k)
		SetEffectiveField(mini.k)
	}

	k := mini.k

	if mini.vel == nil {
		mini.vel = cuda.Buffer(3, size)
		cuda.Madd2(mini.vel, k, k, float32(0.0), float32(0.0))
	}

	vel := mini.vel

	// save original magnetization
	// m0 := cuda.Buffer(3, size)
	// defer cuda.Recycle(m0)
	// data.Copy(m0, m)

	//previous force
	k0 := cuda.Buffer(3, size)
	defer cuda.Recycle(k0)

	vf1 := cuda.Buffer(1, size)
	defer cuda.Recycle(vf1)
	ff1 := cuda.Buffer(1, size)
	defer cuda.Recycle(ff1)

	////calculation energies? here
	gneb := M.Mesh().GNEB_code()
	noi := M.Mesh().NumberOfImages()
	Nz := m.Size()[Z]

	///max torque
	MaxTorq := make([]float32, noi)
	Energy := make([]float32, noi)

	if (gneb == 1) || (gneb == 2) {
		coef := float32(Msat.GetRegion(0)) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z]/noi)
		ReactionCoord := make([]float32, noi)
		
		Distance := make([]float32, noi)
		TangentP := make([]float32, noi)

		//Eenrgies of each image
		en := cuda.Buffer(1, size)
		defer cuda.Recycle(en)

		// k0 := cuda.Buffer(3, size)
		// defer cuda.Recycle(k0)
		data.Copy(k0, k)
		B_ext.AddTo(k0)

		// AddExchangeField(k)

		cuda.AddDotProduct2(en, 1.0, k0, m)
		for i := 0; i < noi; i++ {
			Energy[i] = -0.5 * coef * getReactionCoordinate(en, i, noi) //
		}

		cuda.CrossProduct(k0, m, k0)
		cuda.AddDotProduct2(en, 1.0, k0, k0)
		for i := 0; i < (noi); i++ {
			MaxTorq[i] = getReactionCoordinate(en, i, noi)
		}

		//get max energy
		Pos := 0
		E0 := Energy[Pos]
		for i := 1; i < noi; i++ {
			if Energy[i] > E0 {
				E0 = Energy[i]
				Pos = i
			}
		}
		Saddle = Pos

		// if Saddle > 0 && Saddle < noi {
		// 	Pos = Saddle
		// }

		if DoRelax != 1 {
			//shiftet image for distance and reaction coordinates calculation
			ms := cuda.Buffer(3, size)
			defer cuda.Recycle(ms)
			cuda.ShiftMagZ(ms, m, -(Nz / noi), 0.0, 0.0)

			mms := cuda.Buffer(1, size)
			defer cuda.Recycle(mms)
			cuda.AddDotProduct2(mms, 1.0, m, ms)
			mv := cuda.Buffer(1, size)
			defer cuda.Recycle(mv)
			cuda.GetPhi(mv, mms, 1.0, m, ms)

			md := cuda.Buffer(3, size)
			defer cuda.Recycle(md)
			cuda.Madd2(md, m, ms, 1.0, -1.0)
			cuda.AddDotProduct2(mms, 1.0, md, md)

			ReactionCoord[0] = 0.0
			for i := 1; i < noi; i++ {
				Distance[i-1] = float32(math.Sqrt(float64(getReactionCoordinate(mms, i-1, noi))))
				ReactionCoord[i-1] = float32(math.Sqrt(float64(getReactionCoordinate(mv, i-1, noi))))
			}
			for i := 1; i < (noi - 1); i++ {
				cuda.Tangent(md, m, i, noi, Energy[i-1], Energy[i], Energy[i+1], Distance[i-1], Distance[i])
			}

			// setForce(float32(math.Sqrt(float64(cuda.Dot(md,m))))/float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z]))
			// print("to chto nol = ", float64(cuda.Dot(m,md))/float64(m.Size()[X]*m.Size()[Y]*m.Size()[Z]), "\n")

			// mmss := cuda.Buffer(1, size)
			// defer cuda.Recycle(mmss)
			cuda.AddDotProduct2(mms, 1.0, md, md)
			for i := 1; i < (noi - 1); i++ {
				TangentP[i] = getReactionCoordinate(mms, i, noi)
			}

			//tau magnetization dot
			// cuda.AddDotProduct2(mms,1.0,m,md)
			// for i:=1; i<(noi-1); i++{
			// 	print(i,",", getReactionCoordinate(mms,i,noi), ",", Energy[i], "\n")
			// }

			cuda.AddDotProduct2(mms, 1.0, md, k)
			for i := 1; i < (noi - 1); i++ {
				Distance[i] = getReactionCoordinate(mms, i, noi)
			}

			for i := 1; i < (noi - 1); i++ {
				// cuda.GNEB(k,md,m,i,noi,TangentP[i],Energy[i-1],Energy[i],Energy[i+1],ReactionCoord[i],ReactionCoord[i+1],float32(k_force))
				cuda.GNEB(k, md, m, i, noi, TangentP[i], ReactionCoord[i-1], ReactionCoord[i], float32(k_force), CIGNEB, Saddle)
			}

		}

		if NSteps%WritingIter == 0 {
			// print("Energy[0]=", Energy[0], "Energy[",Pos,"]=", Energy[Pos], "Energy[",noi-1,"]=", Energy[noi-1], "\n")
			if InterpolateEnergy == 1 {

				TangentP[0] = TangentP[1]
				TangentP[noi-1] = TangentP[noi-2]

				for i := 1; i < (noi - 1); i++ {
					TangentP[i] = float32(math.Sqrt(float64(TangentP[i]))) * float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z]/noi) / float32(Msat.GetRegion(0))
					// print( Distance[i]/TangentP[i], "\n")
				}

				C.my_print_interp(C.int(noi), (*C.float)(unsafe.Pointer(&ReactionCoord[0])), (*C.float)(unsafe.Pointer(&TangentP[0])),
					(*C.float)(unsafe.Pointer(&Distance[0])),
					(*C.float)(unsafe.Pointer(&Energy[0])), C.CString(OD()+"table.txt"), (C.float)(2.0*noi))
			} else {
				C.my_print2(C.int(noi), (*C.float)(unsafe.Pointer(&ReactionCoord[0])),
					(*C.float)(unsafe.Pointer(&Distance[0])),
					(*C.float)(unsafe.Pointer(&Energy[0])),
					(*C.float)(unsafe.Pointer(&MaxTorq[0])), C.CString(OD()+"table.txt"))
			}

		}

	}

	torque := float32(1.0)
	if (gneb == 1) || (gneb == 2) {

		if DoRelax != 1 {
			if MinimizeEndPoints == 0 {
				cuda.CopyPath(m0, k, noi)
				torque = float32(math.Sqrt(float64(cuda.Dot(m0, m0)))) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
				setForce(torque)
			} else {
				cuda.CrossProduct(m0, m, k)
				cuda.AddDotProduct2(ff1, 1.0, m0, m0)
				torque = 0.5 * (float32(math.Sqrt(float64(getReactionCoordinate(ff1, 0, noi)))) + float32(math.Sqrt(float64(getReactionCoordinate(ff1, noi-1, noi))))) / float32(m.Size()[X]*m.Size()[Y])
				setForce(torque)
			}

			// print(float32(math.Sqrt(float64(cuda.Dot(k,k))))/float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z]),"\n")
		} else {
			cuda.CrossProduct(m0, m, k)
			torque = float32(math.Sqrt(float64(cuda.Dot(m0, m0)))) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
			setForce(torque)
		}
		// else{
		// 	cuda.CrossProduct(m0, m, k)
		// 	torque = float32(math.Sqrt(float64(cuda.Dot(m0,m0))))/float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
		// 	setForce(torque)
		// }

	} else {
		// setMaxDot(m,k)
		cuda.CrossProduct(m0, m, k)
		torque = float32(math.Sqrt(float64(cuda.Dot(m0, m0)))) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
		// torque = float32(cuda.MaxVecNorm(m0))
		setForce(torque)

		// tv := cuda.Dot(vel,vel)
		// C.test_print(C.float(tv),C.float(torque),C.CString(OD()+"table3.csv"));

	}

	if torque < float32(MaxForce) {
		print("Torque = ", torque, " Number of steps = ", NSteps, "\n")
		//energies
		E0 := Energy[0]
		for i := 1; i < noi; i++ {
			if Energy[i] > E0 {
				E0 = Energy[i]
			}
		}
		print("E_min1 = ", Energy[0], " E_min2 = ", Energy[noi-1], " E_saddle = ", E0, "\n")
		print(" Delta_E1 = ", E0-Energy[0], " Delta_E2 = ", E0-Energy[noi-1], "\n")
		NSteps = MaxIter
	}

	//get energy
	// data.Copy(k0, k)
	// B_ext.AddTo(k0)
	// e1:=cuda.Dot(k0,m)
	// cuda.TurnOnGeom(k,geometry.Gpu());
	///Projection of the force onto the tangent space
	cuda.Projection(k, m)
	// Now we have Dot(k,m) = 0

	// cuda.AddDotProduct2(vf1,1.0,m,k)
	// mk0 := float32(cuda.Dot(vf1,vf1))/float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])

	// Calculation of v
	// if(NSteps == 0) {
	// 	cuda.Zero(vel);
	// }
	data.Copy(k0, vel)
	cuda.Madd2(vel, k0, k, float32(1.0), float32(0.5*stepsize/mass))
	// vf := float32(0.0)

	// tv := cuda.Dot(k,vel)
	// vv := cuda.Dot(vel,vel)
	// C.test_print(C.float(vv),C.float(tv),C.CString(OD()+"table3.csv"));

	// m1 := cuda.Buffer(1, size)
	// defer cuda.Recycle(m1)
	// cuda.AddDotProduct2(m1,1.0,vel,k)

	// m2 := cuda.Buffer(1, size)
	// defer cuda.Recycle(m2)
	// cuda.AddDotProduct2(m2,1.0,k,k)
	// vf = cuda.Sum(m1)
	// ff := cuda.Sum(m2)
	// vf = vf/ff
	// vf := cuda.Dot(vel, k)
	// ff := cuda.Dot(k, k)
	cuda.AddDotProduct2(ff1, 1.0, vel, k)
	vf := getReactionCoordinate(ff1, 0, noi) + getReactionCoordinate(ff1, noi-1, noi)
	cuda.AddDotProduct2(ff1, 1.0, k, k)
	ff := getReactionCoordinate(ff1, 0, noi) + getReactionCoordinate(ff1, noi-1, noi)

	if MinimizeEndPoints == 1 {
		vf = vf / ff
	} else {
		vf = cuda.Dot(vel, k) // - vf
		ff = cuda.Dot(k, k)   // - ff
		vf = vf / ff
	}

	if vf <= 0 {
		// print("tuta step = ", NSteps, " vf/ff = ", vf, "\n")
		// cuda.Zero(v)
		vf = float32(0.0)
	}
	if NSteps == 0 {
		vf = 0.0
	}

	// cuda.GetVelocity(v,k,m1,m2);
	// cuda.Madd2(k0,v,k,float32(1.0),float32(0.5*stepsize/mass))
	// else{
	// data.Copy(v,k)
	// }
	// e2:=cuda.Dot(v,v)
	// vel := vf*float32(math.Sqrt(float64(cuda.Dot(v,v))))/float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
	// vf = float32(0.0)
	///Having v in the tangent space we can get the search direction
	cuda.Madd2(k0, k, k, vf, float32(0.5*stepsize/mass))

	///note for Euler
	// cuda.Madd2(v,v,k,float32(1.0),float32(0.5*stepsize/mass))

	// save the magnetization
	data.Copy(m0, m)

	//and perform one VPO step
	cuda.VPOminimize(m, k0, regions.Gpu(), float32(stepsize), MinimizeEndPoints, noi)
	// cuda.GMinimize(m, k0, float32(stepsize))
	// M.normalize()
	//now we project k0 onto the tangent of new magnetization
	cuda.Velocity(vel, k0, m, m0)
	// data.Copy(v,k0)
	// cuda.Velocity2(v,k0,m,m0, float32(stepsize));
	//now we have Dot(k0,m) = 0

	// cuda.Madd2(v,k0,k0,float32(1.0),float32(0.0))
	//torque

	SetEffectiveField(k)

	// cuda.ZeroMask(k, FrozenSpins.gpuLUT1(), regions.Gpu())

	////testing
	// cuda.CrossProduct(m0, m, k)
	// trq := float32(math.Sqrt(float64(cuda.Dot(m0,m0))))/float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])

	// // k0k0 := float32(math.Sqrt(float64(cuda.Dot(k0,k0))))/float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])

	// C.vpo_print(C.int(NSteps),C.float(trq) ,C.float(e1) ,C.float(0.5*e2), C.float(e1+0.5*e2) , C.CString(OD()+"table.txt"))

	// C.vpo_print(C.int(NSteps),C.float(vel) , C.CString(OD()+"table.txt"))

	/////end testing

	// data.Copy(m0, m)

	// setForce(float32(math.Sqrt(float64(cuda.Dot(k0,k0))))/float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z]))
	// cuda.VPOminimize(m, k0, float32(stepsize))
	// C.vpo_print(C.int(NSteps),C.float(cuda.Dot(k0,k0)) , C.CString(OD()+"table.txt"))

	// C.vpo_print(C.int(NSteps),C.float(float32(math.Sqrt(float64(cuda.Dot(k0,m))))) , C.CString(OD()+"table.txt"))
	// C.vpo_print(C.int(NSteps),C.float(cuda.Dot(k0,m)) , C.CString(OD()+"table.txt"))

	// cuda.CopyPath(m0,v,noi)

	// make descent
	// ff := float32(0.0);
	// if(NSteps<2){
	// 	data.Copy(k0, k)
	// }else{
	// 	forces := cuda.Buffer(3, size)
	// 	defer cuda.Recycle(forces)
	// 	cuda.Madd2(forces,k,k0,float32(0.5*mass*stepsize),float32(0.5*mass*stepsize))
	// 	cuda.Madd2(v,v,forces,1.0,1.0)
	// 	vf := cuda.Dot(v,k)
	// 	if(vf<0){
	// 		cuda.Zero(v)
	// 	}else{
	// 		ff = cuda.Dot(k,k)
	// 		ff = vf/ff;
	// 	}
	// }

	//minimize only if gneb = true
	// pref_force := 0.5*mass*stepsize*stepsize;
	// cuda.VPOminimize(m, k, k, float32(stepsize)*ff,float32(pref_force))

	// data.Copy(k0, k)

	// setForce(float32(math.Sqrt(float64(cuda.Dot(m,k))))/float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z]))

	// k0 := cuda.Buffer(3, size)
	// defer cuda.Recycle(k0)
	// data.Copy(k0, k)
	// torqueFn(k0)

	// setMaxTorque(k0) // report to user

	// as a convention, time does not advance during relax
	NSteps++
}

func (mini *VPOminimizer) Free() {
	mini.k.Free()
	mini.vel.Free()
}

func VPOminimize() {
	NSteps = 0
	SanityCheck()
	// Save the settings we are changing...
	// prevType := solvertype
	// prevFixDt := FixDt
	// prevPrecess := Precess
	// t0 := Time

	relaxing = true // disable temperature noise

	Precess = false // disable precession for torque calculation
	// remove previous stepper
	if stepper != nil {
		stepper.Free()
	}

	// set stepper to the minimizer
	mini := VPOminimizer{
		k:   nil,
		vel: nil}
	stepper = &mini

	cond := func() bool {
		if MaxIter < NSteps {
			return (false)
		} else {
			return (true)
		}
		// return (mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm)

	}

	// cond := true

	RunWhile(cond)
	pause = true
}
