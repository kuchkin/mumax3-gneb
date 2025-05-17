package cuda

import (
	// "unsafe"
	// "fmt"
	"github.com/kuchkin/mumax3-gneb/data"
)

//  Beff to Forces.
// 	m: normalized magnetization
// 	B: effective field in Tesla
// 	Aex_red: Aex / (Msat * 1e18 m2)
// see gneb.cu

func Tangent(T, m *data.Slice, image, noi int, Ep, Ei, En, Lp, Ln float32) {
	N := T.Size()
	cfg := make3DConf(N)

	k_tangent_async(T.DevPtr(X), T.DevPtr(Y), T.DevPtr(Z), m.DevPtr(X),
		m.DevPtr(Y), m.DevPtr(Z), N[X], N[Y], N[Z], noi, image, Ep, Ei, En, Lp, Ln, cfg)

}
