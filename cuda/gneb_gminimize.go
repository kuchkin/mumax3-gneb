package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
)

func GMinimize(m, Beff *data.Slice, dt float32) {
	N := m.Len()
	cfg := make1DConf(N)

	k_gminimize_async(m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		dt, N, cfg)
}
