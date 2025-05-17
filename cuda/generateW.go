package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
)

func GenerateW(w2 *data.Slice) {
	N := w2.Len()
	cfg := make1DConf(N)

	k_generate_w_async(w2.DevPtr(X), w2.DevPtr(Y), N, cfg)
}
