package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
)

func GetHW(k0, k, u1, u2, hw *data.Slice, epsilon float32) {
	N := k0.Len()
	cfg := make1DConf(N)

	k_get_hw_async(k0.DevPtr(X), k0.DevPtr(Y), k0.DevPtr(Z),
		k.DevPtr(X), k.DevPtr(Y), k.DevPtr(Z),
		u1.DevPtr(X), u1.DevPtr(Y), u1.DevPtr(Z),
		u2.DevPtr(X), u2.DevPtr(Y), u2.DevPtr(Z),
		hw.DevPtr(X), hw.DevPtr(Y), N, epsilon, cfg)
}
