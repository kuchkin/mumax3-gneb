package engine

import (
	"github.com/kuchkin/mumax3-gneb/cuda"
	"github.com/kuchkin/mumax3-gneb/data"
	"github.com/kuchkin/mumax3-gneb/util"
	"math"
)

// Classical 4th order RK solver.
type RK4D struct {
	n *data.Slice
}

func (rk *RK4D) Step() {
	m := M.Buffer()
	size := m.Size()

	if FixDt != 0 {
		Dt_si = FixDt
	}

	//initialize 4th component to mx and normalize vector (mx,my,mz,n)
	if rk.n == nil {
		rk.n = cuda.Buffer(1, size)
		M.random(rk.n)
		M.normalize4D(rk.n)
	}
	n := rk.n

	// nn := cuda.Dot(n, n)
	// print("tuta step = ", NSteps, " nn = ", nn, "\n")

	t0 := Time
	// backup magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	n0 := cuda.Buffer(1, size)
	defer cuda.Recycle(n0)
	data.Copy(n0, n)
	

	k1, k2, k3, k4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(k1)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)

	l1, l2, l3, l4 := cuda.Buffer(1, size), cuda.Buffer(1, size), cuda.Buffer(1, size), cuda.Buffer(1, size)
	defer cuda.Recycle(l1)
	defer cuda.Recycle(l2)
	defer cuda.Recycle(l3)
	defer cuda.Recycle(l4)


	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL

	// stage 1
	SetEffectiveField4D(n, l1)
	torqueFn4D(n,l1,k1)

	// stage 2
	Time = t0 + (1./2.)*Dt_si
	cuda.Madd2(m, m, k1, 1, (1./2.)*h) // m = m*1 + k1*h/2
	cuda.Madd2(n, n, l1, 1, (1./2.)*h) 
	M.normalize4D(n)
	SetEffectiveField4D(n, l2)
	torqueFn4D(n,l2,k2)

	// stage 3
	cuda.Madd2(m, m0, k2, 1, (1./2.)*h) // m = m0*1 + k2*1/2
	cuda.Madd2(n, n0, l2, 1, (1./2.)*h)
	M.normalize4D(n)
	SetEffectiveField4D(n, l3)
	torqueFn4D(n,l3,k3)

	// stage 4
	Time = t0 + Dt_si
	cuda.Madd2(m, m0, k3, 1, 1.*h) // m = m0*1 + k3*1
	cuda.Madd2(n, n0, l3, 1, 1.*h) 
	M.normalize4D(n)
	SetEffectiveField4D(n, l4)
	torqueFn4D(n,l4,k4)
	
	

	err := cuda.MaxVecDiff(k1, k4) * float64(h)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		// 4th order solution
		cuda.Madd5(m, m0, k1, k2, k3, k4, 1, (1./6.)*h, (1./3.)*h, (1./3.)*h, (1./6.)*h)
		cuda.Madd5(n, n0, l1, l2, l3, l4, 1, (1./6.)*h, (1./3.)*h, (1./3.)*h, (1./6.)*h)
		M.normalize4D(n)
		NSteps++
		adaptDt(math.Pow(MaxErr/err, 1./4.))
		setLastErr(err)
		setMaxTorque(k4)
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m, m0)
		data.Copy(n, n0)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./5.))
	}
}

func (rk *RK4D) Free() {
}
