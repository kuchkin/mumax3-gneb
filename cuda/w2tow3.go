package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
)

func W2ToW3(u1, u2, w2, w3 *data.Slice) {
	N := u1.Len()
	cfg := make1DConf(N)

	k_w2tow3_async(u1.DevPtr(X), u1.DevPtr(Y), u1.DevPtr(Z),
		u2.DevPtr(X), u2.DevPtr(Y), u2.DevPtr(Z),
		w2.DevPtr(X), w2.DevPtr(Y),
		w3.DevPtr(X), w3.DevPtr(Y), w3.DevPtr(Z),
		N, cfg)
}
