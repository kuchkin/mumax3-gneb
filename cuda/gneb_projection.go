package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
	"github.com/kuchkin/mumax3-gneb/util"
)

func Projection(k, m *data.Slice) {

	util.Argument(k.Len() == m.Len())

	N := k.Len()
	cfg := make1DConf(N)
	k_projection_async(k.DevPtr(X), k.DevPtr(Y), k.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		N, cfg)
}


func Projection4D(k, k4, m, n *data.Slice) {

	util.Argument(k.Len() == m.Len())

	N := k.Len()
	cfg := make1DConf(N)
	k_projection4D_async(k.DevPtr(X), k.DevPtr(Y), k.DevPtr(Z), k4.DevPtr(X),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z), n.DevPtr(X),
		N, cfg)
}