package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
)

func VPOminimize(m, Beff *data.Slice, regions *Bytes, dt float32, minend, noi int) {
	N := m.Len()
	cfg := make1DConf(N)
	NN := Beff.Size()

	k_vpo_async(m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z), regions.Ptr,
		dt, minend, noi, N, NN[Z], cfg)
}
