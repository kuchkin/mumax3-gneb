package engine

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
*/
import "C"

var (
	mass              = 1.0
	MaxForce          = 100.0
	CIGNEB            = 0
	Saddle            = 0
)

func init() {
	DeclFunc("VPOminimize", VPOminimize, "Use VPO method to zeros the forces (or energy gradients)")
	DeclVar("mass", &mass, "mass")
	DeclVar("MaxForce", &MaxForce, "MaxForce")
	DeclVar("CIGNEB", &CIGNEB, "CIGNEB")
	DeclVar("Saddle", &Saddle, "Saddle")
}

type VPOminimizer struct {
	k   *data.Slice // torque saved to calculate time step
	vel *data.Slice // torque saved to calculate time step
}

func (mini *VPOminimizer) Step() {
	m := M.Buffer()
	size := m.Size()

	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)

	if mini.k == nil {
		mini.k = cuda.Buffer(3, size)
		SetEffectiveField(mini.k)
	}

	k := mini.k

	if mini.vel == nil {
		mini.vel = cuda.Buffer(3, size)
		cuda.Madd2(mini.vel, k, k, float32(0.0), float32(0.0))
	}
	vel := mini.vel

	//previous force
	k0 := cuda.Buffer(3, size)
	defer cuda.Recycle(k0)

	vf1 := cuda.Buffer(1, size)
	defer cuda.Recycle(vf1)
	ff1 := cuda.Buffer(1, size)
	defer cuda.Recycle(ff1)

	////GNEB parameters
	gneb 	:= M.Mesh().GNEB_code() 
	noi 	:= M.Mesh().NumberOfImages()
	Nz 		:= m.Size()[Z]

	///max torque
	MaxTorq := make([]float32, noi)
	Energy 	:= make([]float32, noi)

	if (gneb == 1) || (gneb == 2) {
		coef := float32(Msat.GetRegion(0)) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z]/noi)
		ReactionCoord := make([]float32, noi)
		
		Distance := make([]float32, noi)
		TangentP := make([]float32, noi)

		//Eenrgies of each image
		en := cuda.Buffer(1, size)
		defer cuda.Recycle(en)

		data.Copy(k0, k)
		B_ext.AddTo(k0)

		cuda.AddDotProduct2(en, 1.0, k0, m)
		for i := 0; i < noi; i++ {
			Energy[i] = -0.5 * coef * getReactionCoordinate(en, i, noi) 
		}


		cuda.CrossProduct(k0, m, k)
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

			cuda.AddDotProduct2(mms, 1.0, md, md)
			for i := 1; i < (noi - 1); i++ {
				TangentP[i] = getReactionCoordinate(mms, i, noi)
			}

			cuda.AddDotProduct2(mms, 1.0, md, k)
			for i := 1; i < (noi - 1); i++ {
				Distance[i] = getReactionCoordinate(mms, i, noi)
			}

			for i := 1; i < (noi - 1); i++ {
				cuda.GNEB(k, md, m, i, noi, TangentP[i], ReactionCoord[i-1], ReactionCoord[i], float32(k_force), CIGNEB, Saddle)
			}

		}

		if NSteps%WritingIter == 0 {
			C.my_print2(C.int(noi), (*C.float)(unsafe.Pointer(&ReactionCoord[0])),
					(*C.float)(unsafe.Pointer(&Distance[0])),
					(*C.float)(unsafe.Pointer(&Energy[0])),
					(*C.float)(unsafe.Pointer(&MaxTorq[0])), C.CString(OD()+"table.txt"))

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
		} else {
			cuda.CrossProduct(m0, m, k)
			torque = float32(math.Sqrt(float64(cuda.Dot(m0, m0)))) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
			setForce(torque)
		}

	} else {
		cuda.CrossProduct(m0, m, k)
		torque = float32(math.Sqrt(float64(cuda.Dot(m0, m0)))) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
		setForce(torque)
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

	
	///Projection of the force onto the tangent space
	cuda.Projection(k, m)
	// Now we have Dot(k,m) = 0

	

	// Calculation of velocity
	data.Copy(k0, vel)
	cuda.Madd2(vel, k0, k, float32(1.0), float32(0.5*stepsize/mass))

	cuda.AddDotProduct2(ff1, 1.0, vel, k)
	vf := getReactionCoordinate(ff1, 0, noi) + getReactionCoordinate(ff1, noi-1, noi)
	cuda.AddDotProduct2(ff1, 1.0, k, k)
	ff := getReactionCoordinate(ff1, 0, noi) + getReactionCoordinate(ff1, noi-1, noi)

	if MinimizeEndPoints == 1 {
		vf = vf / ff
	} else {
		vf = cuda.Dot(vel, k)
		ff = cuda.Dot(k, k) 
		vf = vf / ff
	}

	if vf <= 0 {
		vf = float32(0.0)
	}
	if NSteps == 0 {
		vf = 0.0
	}

	
	///Having velocity in the tangent space we can get the search direction
	cuda.Madd2(k0, k, k, vf, float32(0.5*stepsize/mass))

	// save the magnetization
	data.Copy(m0, m)

	//and perform one VPO step
	cuda.VPOminimize(m, k0, regions.Gpu(), float32(stepsize), MinimizeEndPoints, noi)
	
	//now we project k0 onto the tangent of new magnetization
	cuda.Velocity(vel, k0, m, m0)
	
	//calculate Beff for the next VPO step
	SetEffectiveField(k)

	NSteps++
}

func (mini *VPOminimizer) Free() {
	mini.k.Free()
	mini.vel.Free()
}

func VPOminimize() {
	NSteps = 0
	SanityCheck()
	
	relaxing = true // disable temperature noise
	Precess = false // disable precession for torque calculation
	// remove previous stepper
	if stepper != nil {
		stepper.Free()
	}

	// set stepper to the VPOminimizer
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
	}

	RunWhile(cond)
	pause = true
}
