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
	k    *data.Slice // torque saved to calculate time step
	vel  *data.Slice // torque saved to calculate time step

	k4   *data.Slice // torque saved to calculate time step
	vel4 *data.Slice // torque saved to calculate time step
}

func (mini *VPOminimizer) Step() {
	m := M.Buffer()
	size := m.Size()

	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)


	//nu_4 if we do RGNEB
	n0 := cuda.Buffer(1, size)
	defer cuda.Recycle(n0)
	n  := cuda.Buffer(1, size)
	defer cuda.Recycle(n)


	//if we do RGNEB
	if Kappa>0 && mini.k4 == nil{
		//initialize nu_4
		// temp := cuda.Dot(n,n);
		// if temp <= 0{
			M.random(n)
			M.normalize4D(n)
		// }
		

		//calculate effective field assoc with nu_4 for the first time
		mini.k4 = cuda.Buffer(1, size)
		SetEffectiveField4D(n, mini.k4)
	}
	k4 := mini.k4;


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

	//if we do RGNEB
	if Kappa>0 && mini.vel4 == nil{
		mini.vel4 = cuda.Buffer(1, size)
		cuda.Madd2(mini.vel4, k4, k4, float32(0.0), float32(0.0))
	}
	vel4 := mini.vel4

	//previous force
	k0 := cuda.Buffer(3, size)
	defer cuda.Recycle(k0)

	k04 := cuda.Buffer(1, size)
	defer cuda.Recycle(k04)

	ff := cuda.Buffer(1, size)
	defer cuda.Recycle(ff)


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

		if Kappa > 0{
			cuda.AddDotProduct1(en, 1.0, k4, n)
		}

		for i := 0; i < noi; i++ {
			Energy[i] = -0.5 * coef * getReactionCoordinate(en, i, noi) 
		}

		if Kappa > 0{
			cuda.AddDotProduct3(en, 1.0, m, n, k, k4)
			for i := 0; i < (noi); i++ {
				MaxTorq[i] = getReactionCoordinate(en, i, noi)
				// MaxTorq[i] = MaxTorq[i]*MaxTorq[i]
			}
		}else{
			cuda.CrossProduct(k0, m, k)
			cuda.AddDotProduct2(en, 1.0, k0, k0)
			for i := 0; i < (noi); i++ {
				MaxTorq[i] = getReactionCoordinate(en, i, noi)
			}
		}

		// cuda.CrossProduct(k0, m, k)
		// 	cuda.AddDotProduct2(en, 1.0, k0, k0)
		// 	for i := 0; i < (noi); i++ {
		// 		MaxTorq[i] = getReactionCoordinate(en, i, noi)
		// 	}
		
		

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

			ns := cuda.Buffer(1, size)
			defer cuda.Recycle(ns)
			if Kappa > 0 {
				cuda.ShiftMagZ4(ns, n, -(Nz / noi), 0.0, 0.0)
			}

			mms := cuda.Buffer(1, size)
			defer cuda.Recycle(mms)
			// cuda.AddDotProduct2(mms, 1.0, m, ms)
			// mv := cuda.Buffer(1, size)
			// defer cuda.Recycle(mv)
			// cuda.GetPhi(mv, mms, 1.0, m, ms)

			md := cuda.Buffer(3, size)
			defer cuda.Recycle(md)
			cuda.Madd2(md, m, ms, 1.0, -1.0)
			cuda.AddDotProduct2(mms, 1.0, md, md)

			if Kappa > 0{
				cuda.Madd2(k04, n, ns, 1.0, -1.0)
				cuda.AddDotProduct1(mms, 1.0, k04, k04)
			}


			ReactionCoord[0] = 0.0
			for i := 1; i < noi; i++ {
				Distance[i-1] = float32(math.Sqrt(float64(getReactionCoordinate(mms, i-1, noi))))
				ReactionCoord[i-1] = Distance[i-1];
				//ReactionCoord[i-1] = float32(math.Sqrt(float64(getReactionCoordinate(mv, i-1, noi))))
				// if NSteps == 1{
				// 	print(Distance[i-1], ",", ReactionCoord[i-1], "\n")
				// }
			}
			for i := 1; i < (noi - 1); i++ {
				if Kappa > 0{
					cuda.Tangent4D(md, k04, m, n, i, noi, Energy[i-1], Energy[i], Energy[i+1], Distance[i-1], Distance[i])
				}else{
					cuda.Tangent(md, m, i, noi, Energy[i-1], Energy[i], Energy[i+1], Distance[i-1], Distance[i])
				}
				
			}

			cuda.AddDotProduct2(mms, 1.0, md, md)
			if Kappa > 0{
				cuda.AddDotProduct1(mms, 1.0, k04, k04)
			}
			for i := 1; i < (noi - 1); i++ {
				TangentP[i] = getReactionCoordinate(mms, i, noi)
			}

			///metka - neponiatno zachem
			cuda.AddDotProduct2(mms, 1.0, md, k)
			for i := 1; i < (noi - 1); i++ {
				Distance[i] = getReactionCoordinate(mms, i, noi)
			}

			for i := 1; i < (noi - 1); i++ {
				if Kappa > 0{
					cuda.RGNEB(k, k4, md, k04, m, n, i, noi, TangentP[i], ReactionCoord[i-1], ReactionCoord[i], float32(k_force), CIGNEB, Saddle)
				}else{
					cuda.GNEB(k, md, m, i, noi, TangentP[i], ReactionCoord[i-1], ReactionCoord[i], float32(k_force), CIGNEB, Saddle)
				}
				
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
				cuda.AddDotProduct2(ff, 1.0, m0, m0)
				torque = 0.5 * (float32(math.Sqrt(float64(getReactionCoordinate(ff, 0, noi)))) + float32(math.Sqrt(float64(getReactionCoordinate(ff, noi-1, noi))))) / float32(m.Size()[X]*m.Size()[Y])
				setForce(torque)
			}
		} else {
			if Kappa>0{
				cuda.AddDotProduct3(ff, 1.0, m, n, k, k4)
				torque = float32(math.Sqrt(float64(cuda.Sum(ff)))) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
				setForce(torque)
			}else{
				cuda.CrossProduct(m0, m, k)
				torque = float32(math.Sqrt(float64(cuda.Dot(m0, m0)))) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
				setForce(torque)
			}
			
		}

	} else {
		if Kappa>0{
			cuda.AddDotProduct3(ff, 1.0, m, n, k, k4)
			torque = float32(math.Sqrt(float64(cuda.Sum(ff)))) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
			setForce(torque)
			// // cuda.CrossProduct(m0, m, k)
			// torque = float32(math.Sqrt(float64(cuda.Dot(m0, m0)))) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
			// setForce(torque)
		} else{
			cuda.CrossProduct(m0, m, k)
			torque = float32(math.Sqrt(float64(cuda.Dot(m0, m0)))) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
			setForce(torque)
		}
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

	// if NSteps%WritingIter == 0{
	// 	print((cuda.Dot(m,m)) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z]),"\n")
	// }
	///Projection of the force onto the tangent space
	// cuda.Projection(k, m)
	// Now we have Dot(k,m) = 0
	//if we do RGNEB
	if Kappa > 0 {
		cuda.Projection4D(k, k4, m, n)
		// print(cuda.Dot(k,m) + cuda.Dot(k4,n), "\n")
		///***************************
		// torque = float32(math.Sqrt(float64(cuda.Dot(k, k)+cuda.Dot(k4, k4)))) / float32(m.Size()[X]*m.Size()[Y]*m.Size()[Z])
		// setForce(torque)
	}else{
		cuda.Projection(k, m)
		// print(cuda.Dot(k,m), "\n")
	}

	
	

	// Calculation of velocity
	data.Copy(k0, vel)
	cuda.Madd2(vel, k0, k, float32(1.0), float32(0.5*stepsize/mass))

	if Kappa > 0{
		data.Copy(k04, vel4)
		cuda.Madd2(vel4, k04, k4, float32(1.0), float32(0.5*stepsize/mass))
	}

	cuda.AddDotProduct2(ff, 1.0, vel,  k)
	if Kappa > 0{	
		cuda.AddDotProduct1(ff, 1.0, vel4, k4)
	}
	vf := getReactionCoordinate(ff, 0, noi) + getReactionCoordinate(ff, noi-1, noi)
	// vf1 := getReactionCoordinate(ff, 0, noi);
	// vf2 := 0.0;
	// vf3 := getReactionCoordinate(ff, noi-1, noi)
	

	cuda.AddDotProduct2(ff, 1.0, k, k)
	if Kappa > 0{
		cuda.AddDotProduct1(ff, 1.0, k4, k4)
	}
	fff := getReactionCoordinate(ff, 0, noi) + getReactionCoordinate(ff, noi-1, noi)
	// ff1 := getReactionCoordinate(ff, 0, noi);
	// ff2 := 0.0;
	// ff3 := getReactionCoordinate(ff, noi-1, noi)

	if MinimizeEndPoints == 1 {
		vf = vf / fff
	} else {
		if Kappa > 0{
			vf  = cuda.Dot(vel, k) + cuda.Dot(vel4, k4)
			fff = cuda.Dot(k, k)   + cuda.Dot(k4, k4)
		}else{
			vf = cuda.Dot(vel, k)
			fff = cuda.Dot(k, k)
		}
		vf = vf / fff
	}

	if vf <= 0 {
		vf = float32(0.0)
	}
	// if vf2 <= 0 {
	// 	vf2 = float32(0.0)
	// }
	// if vf3 <= 0 {
	// 	vf3 = float32(0.0)
	// }
	// if NSteps == 0 {
	// 	vf1 = 0.0
	// 	vf2 = 0.0
	// 	vf3 = 0.0
	// }

	// vf1 := cuda.Dot(vel, k); ff1 := float32(0.0)
	// cuda.AddDotProduct2(ff, 1.0, vel,  k)
	// if Kappa > 0{	
	// 	cuda.AddDotProduct1(ff, 1.0, vel4, k4)
	// }
	// vf_temp := getReactionCoordinate(ff, 0, noi) + getReactionCoordinate(ff, noi-1, noi);
	// if Kappa > 0{
	// 	vf1 += cuda.Dot(vel4, k4);
	// }
	// vf1 = vf1 - vf_temp;

	// if vf1 <= 0{
	// 	vf1 = float32(0.0)
	// }else{
	// 	ff1 = cuda.Dot(k, k);
	// 	if Kappa > 0{
	// 		ff1 += cuda.Dot(k4, k4);
	// 	}
	// 	cuda.AddDotProduct2(ff, 1.0, k,  k)
	// 	if Kappa > 0{	
	// 		cuda.AddDotProduct1(ff, 1.0, k4, k4)
	// 	}
	// 	vf_temp := getReactionCoordinate(ff, 0, noi) + getReactionCoordinate(ff, noi-1, noi);
	// 	ff1 = ff1 - vf_temp;

	// 	vf1 = vf1/ff1
	// }
	
	///Having velocity in the tangent space we can get the search direction
	// cuda.Madd2(k0, k, k, vf, float32(0.5*stepsize/mass))
	// if Kappa > 0{
	// 	cuda.Madd2(k04, k4, k4, vf, float32(0.5*stepsize/mass))
	// }

	vf += float32(0.5*stepsize/mass);
	

	// save the magnetization
	data.Copy(m0, m)

	if Kappa > 0{
		data.Copy(n0, n)
	}

	//and perform one VPO step
	if Kappa > 0{
		cuda.VPOminimize4D(m, n, k, k4, regions.Gpu(), float32(stepsize), vf, MinimizeEndPoints, noi)
	}else{
		cuda.VPOminimize(m, k, regions.Gpu(), float32(stepsize), vf, MinimizeEndPoints, noi)
	}
	//now we project k0 onto the tangent of new magnetization
	if Kappa > 0{
		cuda.Velocity4D(vel, vel4, k, k4, m, n, m0, n0)
	}else{
		cuda.Velocity(vel, k, m, m0)
	}
	
	
	//calculate Beff for the next VPO step
	SetEffectiveField(k) 
	if Kappa > 0{
		SetEffectiveField4D(n, k4)
	} 

	NSteps++
}

func (mini *VPOminimizer) Free() {
	mini.k.Free()
	mini.vel.Free()
	mini.k4.Free()
	mini.vel4.Free()
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
		k:    nil,
		vel:  nil,
		k4:   nil,
		vel4: nil}
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
