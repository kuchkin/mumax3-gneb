package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
)

func TurnOnGeom(Beff, vol *data.Slice) {
	N := Beff.Len()
	cfg := make1DConf(N)
	NN := Beff.Size()

	k_geom_vpo_async(Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z), vol.DevPtr(0),
		N, NN[Z], cfg)
}
