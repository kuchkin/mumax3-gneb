package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
)

func Rotate(s, v, st, w3 *data.Slice, epsilon float32) {
	N := w3.Len()
	cfg := make1DConf(N)

	k_rotate_async(s.DevPtr(X), s.DevPtr(Y), s.DevPtr(Z),
		v.DevPtr(X), v.DevPtr(Y), v.DevPtr(Z),
		st.DevPtr(X), st.DevPtr(Y), st.DevPtr(Z),
		w3.DevPtr(X), w3.DevPtr(Y), w3.DevPtr(Z), epsilon,
		N, cfg)
}
