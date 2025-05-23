package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
)

// Add Zhang-Li ST torque (Tesla) to torque.
// see zhangli.cu
func AddZhangLiTorque4D(n,l,torque, m *data.Slice, Msat, J, alpha, xi, pol MSlice, mesh *data.Mesh) {
	c := mesh.CellSize()
	N := mesh.Size()
	cfg := make3DConf(N)

	k_addzhanglitorque4D_async(n.DevPtr(X), l.DevPtr(X),
		torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		J.DevPtr(X), J.Mul(X),
		J.DevPtr(Y), J.Mul(Y),
		J.DevPtr(Z), J.Mul(Z),
		alpha.DevPtr(0), alpha.Mul(0),
		xi.DevPtr(0), xi.Mul(0),
		pol.DevPtr(0), pol.Mul(0),
		float32(c[X]), float32(c[Y]), float32(c[Z]),
		N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}
