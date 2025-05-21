package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
	"github.com/kuchkin/mumax3-gneb/util"
)

// dst += prefactor * dot(a, b), as used for energy density
func AddDotProduct2(dst *data.Slice, prefactor float32, a, b *data.Slice) {
	util.Argument(dst.NComp() == 1 && a.NComp() == 3 && b.NComp() == 3)
	util.Argument(dst.Len() == a.Len() && dst.Len() == b.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	k_dotproduct2_async(dst.DevPtr(0), prefactor,
		a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z),
		b.DevPtr(X), b.DevPtr(Y), b.DevPtr(Z),
		N, cfg)
}

func AddDotProduct1(dst *data.Slice, prefactor float32, a, b *data.Slice) {
	util.Argument(dst.NComp() == 1 && a.NComp() == 1 && b.NComp() == 1)
	util.Argument(dst.Len() == a.Len() && dst.Len() == b.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	k_dotproduct1_async(dst.DevPtr(0), prefactor,
		a.DevPtr(X), b.DevPtr(X), N, cfg)
}

//for torque S3
func AddDotProduct3(dst *data.Slice, prefactor float32, a, a4, b, b4 *data.Slice) {
	util.Argument(dst.NComp() == 1 && a4.NComp() == 1 && b4.NComp() == 1 && a.NComp() == 3 && b.NComp() == 3)
	util.Argument(dst.Len() == a.Len() && dst.Len() == b.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	k_dotproduct3_async(dst.DevPtr(0), prefactor,
		a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z), a4.DevPtr(X),
		b.DevPtr(X), b.DevPtr(Y), b.DevPtr(Z), b4.DevPtr(X), N, cfg)
}
