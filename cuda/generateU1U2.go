package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
)

func GenerateU1U2(m, u1, u2 *data.Slice) {
	N := m.Len()
	cfg := make1DConf(N)

	k_generate_u1u2_async(m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		u1.DevPtr(X), u1.DevPtr(Y), u1.DevPtr(Z), u2.DevPtr(X), u2.DevPtr(Y), u2.DevPtr(Z),
		N, cfg)
}
