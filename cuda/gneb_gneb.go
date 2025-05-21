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

func GNEB(B, T, m *data.Slice, image, noi int, tp, Lp, Ln, k float32, CIGNEB, Pos int) {
	N := B.Size()
	cfg := make3DConf(N)

	k_gneb_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z), T.DevPtr(X), T.DevPtr(Y), T.DevPtr(Z), m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z), N[X], N[Y], N[Z], noi, image, tp, Lp, Ln, k, CIGNEB, Pos, cfg)

}

func RGNEB(B,k4, T,T4, m,n *data.Slice, image, noi int, tp, Lp, Ln, k float32, CIGNEB, Pos int) {
	N := B.Size()
	cfg := make3DConf(N)

	k_rgneb_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z), k4.DevPtr(X),
				  T.DevPtr(X), T.DevPtr(Y), T.DevPtr(Z), T4.DevPtr(X),
				  m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z), n.DevPtr(X),
				  N[X], N[Y], N[Z], noi, image, tp, Lp, Ln, k, CIGNEB, Pos, cfg)

}
