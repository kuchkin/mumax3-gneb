package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
	"github.com/kuchkin/mumax3-gneb/util"
)

// dst += prefactor * dot(a, b), as used for energy density
func GetPhi(dst, src *data.Slice, prefactor float32, a, b *data.Slice) {
	util.Argument(dst.NComp() == 1 && a.NComp() == 3 && b.NComp() == 3)
	util.Argument(dst.Len() == a.Len() && dst.Len() == b.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	k_getphi_async(dst.DevPtr(0), src.DevPtr(0), prefactor,
		a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z),
		b.DevPtr(X), b.DevPtr(Y), b.DevPtr(Z),
		N, cfg)
}
