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
char shortBufer[100];
void my_print(int noi, float *coord,float *dist,float *energy, char *outputfilename) {
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
    	snprintf(shortBufer,100,"%i, %0.15f, %0.15f, %0.15f, %0.15f\n",i,prev,distance,energy[i],coord[i]);
		fputs (shortBufer,pFile);
    }
    fclose (pFile);
}
*/
import "C"

var (
	MaxIter           = 1
	WritingIter       = 1
	k_force           = 0.001
	stepsize          = 0.0001
	DoRelax           = 0
	MinimizeFirst     = 1
	MinimizeLast      = 1
	MinimizeEndPoints = 0
)

func init() {
	DeclFunc("GMinimize", GMinimize, "Use gradient descent method to zeros the forces (or energy gradients)")
	DeclVar("MaxIter", &MaxIter, "MaxIter")
	DeclVar("WritingIter", &WritingIter, "WritingIter")
	DeclVar("k_force", &k_force, "k_force")
	DeclVar("stepsize", &stepsize, "stepsize")
	DeclVar("MinimizeFirst", &MinimizeFirst, "MinimizeFirst")
	DeclVar("DoRelax", &DoRelax, "DoRelax")
	DeclVar("MinimizeLast", &MinimizeLast, "MinimizeLast")
	DeclVar("MinimizeEndPoints", &MinimizeEndPoints, "MinimizeEndPoints")
	// DeclVar("GMinimizerStop", &GStopMaxDm, "Stopping max dM for Minimize")
	// DeclVar("GMinimizerSamples", &GDmSamples, "Number of max dM to collect for Minimize convergence check.")
}

type GMinimizer struct {
	k *data.Slice // torque saved to calculate time step

}

func (mini *GMinimizer) Step() {
	m := M.Buffer()
	size := m.Size()

	if mini.k == nil {
		mini.k = cuda.Buffer(3, size)
		// torqueFn(mini.k)
		SetEffectiveField(mini.k)
	}

	k := mini.k

	// save original magnetization
	// m0 := cuda.Buffer(3, size)
	// defer cuda.Recycle(m0)
	// data.Copy(m0, m)

	////calculation energies? here
	gneb := M.Mesh().GNEB_code()
	noi := M.Mesh().NumberOfImages()
	Nz := m.Size()[Z]

	if (gneb == 1) || (gneb == 2) {
		coef := float32(Msat.GetRegion(0)) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z]/noi)
		ReactionCoord := make([]float32, noi)
		Energy := make([]float32, noi)
		Distance := make([]float32, noi)
		TangentP := make([]float32, noi)

		//Eenrgies of each image
		en := cuda.Buffer(1, size)
		defer cuda.Recycle(en)

		k0 := cuda.Buffer(3, size)
		defer cuda.Recycle(k0)
		data.Copy(k0, k)
		B_ext.AddTo(k0)

		cuda.AddDotProduct2(en, 1.0, k0, m)
		for i := 0; i < noi; i++ {
			Energy[i] = -0.5 * coef * getReactionCoordinate(en, i, noi)
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

			// mmss := cuda.Buffer(1, size)
			// defer cuda.Recycle(mmss)
			cuda.AddDotProduct2(mms, 1.0, md, md)
			for i := 1; i < (noi - 1); i++ {
				TangentP[i] = getReactionCoordinate(mms, i, noi)
			}

			for i := 1; i < (noi - 1); i++ {
				// cuda.GNEB(k,md,m,i,noi,TangentP[i],Energy[i-1],Energy[i],Energy[i+1],ReactionCoord[i],ReactionCoord[i+1],float32(k_force))
				cuda.GNEB(k, md, m, i, noi, TangentP[i], ReactionCoord[i-1], ReactionCoord[i], float32(k_force), CIGNEB, Pos)
			}

		}

		if NSteps%WritingIter == 0 {
			// print("Energy[0]=", Energy[0], "Energy[",Pos,"]=", Energy[Pos], "Energy[",noi-1,"]=", Energy[noi-1], "\n")
			C.my_print(C.int(noi), (*C.float)(unsafe.Pointer(&ReactionCoord[0])),
				(*C.float)(unsafe.Pointer(&Distance[0])),
				(*C.float)(unsafe.Pointer(&Energy[0])), C.CString(OD()+"table.txt"))
		}

	}

	// make descent

	//minimize only if gneb = true
	torque := float32(1.0)
	cuda.GMinimize(m, k, float32(stepsize))

	if ((gneb == 1) || (gneb == 2)) && MinimizeEndPoints == 0 {

		// fs  := cuda.Buffer(1, size)
		// defer cuda.Recycle(fs)
		// cuda.AddDotProduct2(fs,1.0,k,k)
		// setTotalForce(fs,noi)
		if DoRelax != 1 {
			cuda.CopyPath(k, k, noi)
			torque = float32(math.Sqrt(float64(cuda.Dot(k, k)))) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
			setForce(torque)
		}
		// else{
		// 	cuda.CrossProduct(k, m, k)
		// 	torque = float32(math.Sqrt(float64(cuda.Dot(k,k))))/float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
		// 	setForce(torque)

		// }
	} else {
		// setMaxDot(m,k)
		cuda.CrossProduct(k, m, k)
		torque = float32(math.Sqrt(float64(cuda.Dot(k, k)))) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
		setForce(torque)
	}
	if torque < float32(MaxForce) {
		print("Number of steps = ", NSteps, "\n")
		NSteps = MaxIter
	}
	SetEffectiveField(k)
	// k0 := cuda.Buffer(3, size)
	// defer cuda.Recycle(k0)
	// data.Copy(k0, k)
	// torqueFn(k)

	// setMaxTorque(k) // report to user

	// as a convention, time does not advance during relax
	NSteps++
}

func (mini *GMinimizer) Free() {
	mini.k.Free()
}

func GMinimize() {
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
	mini := GMinimizer{
		k: nil}
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
