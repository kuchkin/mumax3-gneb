package engine

// Eigenvalues calculation using Lanczos method

import (
	"github.com/kuchkin/mumax3-gneb/cuda"
	"github.com/kuchkin/mumax3-gneb/data"
	"math"
	"os"
	"unsafe"
)

/*
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#define ROTATE(a,i,j,k,l,tau,s) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);\
a[k][l]=h+s*(g-h*tau);
char shortBuferEig[100];
void print_eig(int Nsub, float *values, char *outputfilename) {
	FILE * pFile;
	pFile = fopen (outputfilename,"w");
    if (pFile!=NULL){
    }
    for(int i = 0;i<Nsub; i++){
    	snprintf(shortBuferEig,100,"%i, %0.15f\n",i,values[i]);
		fputs (shortBuferEig,pFile);
    }
    fclose (pFile);
}
float root(float a){
	float x = 1.0;
	float dif = 1.0;
	int MaxInt = 10000;
	int ind = 0;
	while(dif>0.000001 && ind<MaxInt){
		dif = x;
		x = 0.5*(x+a/x);
		dif -= x;
		dif = fabs(dif);
		ind++;
	}
	if(ind == MaxInt) x = 0.5*a;
	return x;
}
void Jacobi(int Nsub, float *alpha, float *beta) {
	float* a[Nsub];
    for (int i = 0; i < Nsub; i++) a[i] = (float*)malloc(Nsub * sizeof(float));
    for(int i = 0; i<Nsub; i++){
        for(int j = 0; j<Nsub; j++){
            a[i][j] = 0.0;
        }
        a[i][i] = alpha[i];
        if(i!=(Nsub-1)) a[i][i+1] = beta[i];
        if(i!=(Nsub-1)) a[i+1][i] = beta[i];
    }

    //jacobi algorithm
    int j,iq,ip,i, nrot = 0, n = Nsub;
    float tresh,theta,tau,t,sm,s,h,g,c,*b,*z;
    b = (float*)malloc(Nsub * sizeof(float));
    z = (float*)malloc(Nsub * sizeof(float));

    int MaxNum = 10*Nsub;

        for (ip=0;ip<n;ip++) {
        b[ip]=alpha[ip]=a[ip][ip];
        z[ip]=0.0;
    }
    for (i=1;i<=MaxNum;i++) {
        sm=0.0;
        for (ip=0;ip<(n-1);ip++) {
            for (iq=ip+1;iq<n;iq++)
            sm += fabs(a[ip][iq]);
        }
        if (sm == 0.0) {
            i = MaxNum;
            break;
        }
        if(i<3){
            tresh = 0.2*sm/(1.0*n*n);
        }else{
            tresh = 0.0;
        }
        for (ip=0;ip<(n-1);ip++) {
            for (iq=ip+1;iq<n;iq++){
                g = 100.0*fabs(a[ip][iq]);
                if (i > 4 && (float)(fabs(alpha[ip])+g) == (float)fabs(alpha[ip])
                && (float)(fabs(alpha[iq])+g) == (float)fabs(alpha[iq])){
                    a[ip][iq]=0.0;
                }else if(fabs(a[ip][iq])>tresh){
                    h = alpha[iq]-alpha[ip];
                    if((float)(fabs(h)+g) == (float)fabs(h)){
                        t = 1.0*(a[ip][iq])/h;
                    }else{
                        theta=0.5*h/(a[ip][iq]);
                        t=1.0/(fabs(theta)+root(1.0+theta*theta));
                        if (theta < 0.0) t = -t;
                    }
                    c=1.0/root(1+t*t);
                    s=t*c;
                    tau=s/(1.0+c);
                    h=t*a[ip][iq];
                    z[ip] -= h;
                    z[iq] += h;
                    alpha[ip] -= h;
                    alpha[iq] += h;
                    a[ip][iq]=0.0;
                    for (j=0;j<=ip-1;j++) {
                        ROTATE(a,j,ip,j,iq,tau,s)
                    }
                    for (j=ip+1;j<=iq-1;j++) {
                        ROTATE(a,ip,j,j,iq,tau,s)
                    }
                    for (j=iq+1;j<n;j++) {
                        ROTATE(a,ip,j,iq,j,tau,s)
                    }

                }


            }
        }
        for (ip=0;ip<n;ip++) {
            b[ip] += z[ip];
            alpha[ip]=b[ip];
            z[ip]=0.0;
        }
    }
    for (int i = 0; i < Nsub; i++)
        free(a[i]);

    //sorting
    int k;
   	float p;
    for (i=0;i<n;i++) {
        p=alpha[k=i];
        for (j=i+1;j<n;j++)
	    if (alpha[j] >= p) p=alpha[k=j];
	    if (k != i) {
	        alpha[k]=alpha[i];
	        alpha[i]=p;
	    }
   	}


}
*/
import "C"

var (
	Nsub = 10
)

func init() {
	DeclFunc("EigSolve", EigSolve, "Eigenvalues calculation using Lanczos method")
	DeclVar("Nsub", &Nsub, "Nsub")
}

type Eig struct {
	k0 *data.Slice // initial effective field
	m0 *data.Slice // initial magnetization

	u1 *data.Slice // projection operator
	u2 *data.Slice // projection operator

	w2 *data.Slice // random unit 2N vector
	// w3 *data.Slice // corresponding 3N vector

	v1 *data.Slice // unit 2N vector
	v0 *data.Slice // unit 2N vector

	ALPHA []float32
	BETA  []float32
}

func (mini *Eig) Step() {

	//diagonal and off-diagonal elements
	// ALPHA := make([]float32, Nsub)
	// BETA  := make([]float32, Nsub-1)

	// m0 := M.Buffer()

	if mini.ALPHA == nil {
		mini.ALPHA = make([]float32, Nsub)
	}
	ALPHA := mini.ALPHA

	if mini.BETA == nil {
		mini.BETA = make([]float32, Nsub-1)
	}
	BETA := mini.BETA

	m := M.Buffer()
	size := m.Size()

	if mini.m0 == nil {
		mini.m0 = cuda.Buffer(3, size)
		// mini.m0 = M.Buffer()
		cuda.Madd2(mini.m0, m, m, float32(1.0), float32(0.0))
		// print(NSteps,"\n")
	}
	m0 := mini.m0

	if mini.k0 == nil {
		mini.k0 = cuda.Buffer(3, size)
		SetEffectiveField(mini.k0)
		cuda.Projection(mini.k0, m0)

		// print(NSteps,"\n")
	}
	k0 := mini.k0

	// ALPHA[0] = cuda.Dot(k0,m)

	if mini.u1 == nil {
		mini.u1 = cuda.Buffer(3, size)
		mini.u2 = cuda.Buffer(3, size)
		cuda.GenerateU1U2(m0, mini.u1, mini.u2)
	}

	u1 := mini.u1
	u2 := mini.u2

	if mini.w2 == nil {
		mini.w2 = cuda.Buffer(2, size)
		cuda.GenerateW(mini.w2)
	}

	w2 := mini.w2

	// if mini.w3 == nil{
	// 	mini.w3 = cuda.Buffer(3, size)
	// 	cuda.W2ToW3(u1,u2,w2,mini.w3)
	// }

	// w3 := mini.w3

	// print(ALPHA[NSteps],"\n")

	if mini.v0 == nil {
		mini.v0 = cuda.Buffer(2, size)
	}
	v0 := mini.v0
	if mini.v1 == nil {
		mini.v1 = cuda.Buffer(2, size)
	}
	v1 := mini.v1

	// m := cuda.Buffer(3, size)
	// defer cuda.Recycle(m)

	if NSteps == 0 {
		w3 := cuda.Buffer(3, size)
		defer cuda.Recycle(w3)
		cuda.W2ToW3(u1, u2, w2, w3)

		// print(cuda.Dot(m0,k0),"\n")
		// vv := cuda.Buffer(3, size)
		// defer cuda.Recycle(vv)
		// cuda.Madd2(vv,m0,m0,float32(1.0),float32(0.0))
		cuda.Rotate(m0, m0, m, w3, float32(stepsize))
		// cuda.Madd2(m,vv,vv,float32(1.0),float32(0.0))
		// print(cuda.Dot(m0,k0),"\n")

		k := cuda.Buffer(3, size)
		defer cuda.Recycle(k)
		SetEffectiveField(k)

		cuda.Projection(k, m)
		cuda.Rotate(m0, k, k, w3, float32(-1.0*stepsize))

		hw := cuda.Buffer(2, size)
		defer cuda.Recycle(hw)
		cuda.GetHW(k0, k, u1, u2, hw, float32(stepsize))

		ALPHA[NSteps] = cuda.Dot(w2, hw)

		// print(cuda.Dot(k,k),"\n")

		cuda.CopyToSubspace(v0, v1, w2, hw, 0, ALPHA[0], BETA[0])

		mini.v0 = v0
		mini.v1 = v1
		mini.w2 = w2

	} else {
		// print(NSteps," ,",cuda.Dot(w2,w2),"\n")

		// print(cuda.Dot(k,k),"\n")

		BETA[NSteps-1] = float32(math.Sqrt(float64(cuda.Dot(w2, w2))))

		if BETA[NSteps-1] == 0.0 {
			print("Gramm-Shmidt needed!\n")
			os.Exit(3)
		}

		cuda.Madd2(w2, w2, w2, float32(0.0), float32(1.0/BETA[NSteps-1]))

		w3 := cuda.Buffer(3, size)
		defer cuda.Recycle(w3)
		cuda.W2ToW3(u1, u2, w2, w3)
		cuda.Rotate(m0, m0, m, w3, float32(stepsize))
		k := cuda.Buffer(3, size)
		defer cuda.Recycle(k)
		SetEffectiveField(k)

		cuda.Projection(k, m)
		cuda.Rotate(m0, k, k, w3, float32(-1.0*stepsize))

		hw := cuda.Buffer(2, size)
		defer cuda.Recycle(hw)
		cuda.GetHW(k0, k, u1, u2, hw, float32(stepsize))

		ALPHA[NSteps] = cuda.Dot(w2, hw)

		cuda.CopyToSubspace(v0, v1, w2, hw, 1, ALPHA[NSteps], BETA[NSteps-1])

		mini.v0 = v0
		mini.v1 = v1
		mini.w2 = w2
		// BETA[NSteps] = float32(math.Sqrt(float64(cuda.Dot(w2,w2))))
		// // print(BETA[i-1],", ")

		// cuda.Madd2(w2, w2, w2, float32(0.0), float32(1.0/BETA[NSteps]))
		// mini.w2 = w2

	}

	// print(NSteps,"\n");

	if NSteps == (Nsub - 1) {
		print("Tridiagonal matrix is obtaied!\n")
		print("Start eigenvalues calculation...\n")
		C.Jacobi(C.int(Nsub), (*C.float)(unsafe.Pointer(&ALPHA[0])), (*C.float)(unsafe.Pointer(&BETA[0])))
		print("Done!\n")
		C.print_eig(C.int(Nsub), (*C.float)(unsafe.Pointer(&ALPHA[0])), C.CString(OD()+"eigenvalues.txt"))
		NSteps = Nsub + 1
		//  for i := 0; i < Nsub; i++ {
		// 	print(ALPHA[i],"\n")
		// }
		// print("\n")
		//  for i := 0; i < Nsub-1; i++ {
		// 	print(BETA[i],"\n")
		// }
	}
	NSteps++

}

func (mini *Eig) Free() {
	mini.k0.Free()
	mini.m0.Free()
	mini.u1.Free()
	mini.u2.Free()
	mini.w2.Free()
	// mini.w3.Free()
	mini.v1.Free()
	mini.v0.Free()
}

func EigSolve() {
	SanityCheck()

	relaxing = true // disable temperature noise

	Precess = false // disable precession for torque calculation
	// remove previous stepper
	if stepper != nil {
		stepper.Free()
	}

	// set stepper to the minimizer
	mini := Eig{
		k0: nil,
		m0: nil,
		u1: nil,
		u2: nil,
		w2: nil,
		// w3:  nil,
		v1:    nil,
		v0:    nil,
		ALPHA: nil,
		BETA:  nil}
	stepper = &mini

	cond := func() bool {
		if Nsub < NSteps {
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
