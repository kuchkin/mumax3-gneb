package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
	// "github.com/kuchkin/mumax3-gneb/util"
)

// dst += prefactor * dot(a, b), as used for energy density
func CopyToSubspace(v0, v1, w2, hw *data.Slice, id int, alpha, beta float32) {

	N := v0.Len()
	cfg := make1DConf(N)
	k_copy_to_subspace_async(v0.DevPtr(X), v0.DevPtr(Y),
		v1.DevPtr(X), v1.DevPtr(Y),
		w2.DevPtr(X), w2.DevPtr(Y),
		hw.DevPtr(X), hw.DevPtr(Y),
		N, id, alpha, beta, cfg)
}
