package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
)

func Velocity(v, k0, m, m0 *data.Slice) {
	N := m.Len()
	cfg := make1DConf(N)
	NN := m.Size()

	k_velocity_async(v.DevPtr(X), v.DevPtr(Y), v.DevPtr(Z),
		k0.DevPtr(X), k0.DevPtr(Y), k0.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		m0.DevPtr(X), m0.DevPtr(Y), m0.DevPtr(Z),
		N, NN[Z], cfg)
}
