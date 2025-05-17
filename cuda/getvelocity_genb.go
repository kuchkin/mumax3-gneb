package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
)

func GetVelocity(v, k, m1, m2 *data.Slice) {
	N := v.Len()
	cfg := make1DConf(N)
	NN := v.Size()

	k_get_velocity_async(v.DevPtr(X), v.DevPtr(Y), v.DevPtr(Z),
		k.DevPtr(X), k.DevPtr(Y), k.DevPtr(Z),
		m1.DevPtr(X), m2.DevPtr(X),
		N, NN[Z], cfg)
}
