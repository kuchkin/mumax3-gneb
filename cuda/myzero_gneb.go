package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
)

// dst += prefactor * dot(a, b), as used for energy density
func MyZero(dst *data.Slice) {
	// util.Argument(dst.NComp() == 1 && a.NComp() == 3 && b.NComp() == 3)
	// util.Argument(dst.Len() == a.Len() && dst.Len() == b.Len())
	NN := dst.Size()
	N := dst.Len()
	cfg := make1DConf(N)
	k_myzero_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		NN[X], NN[Y], NN[Z], cfg)
}
