package cuda

import (
	"github.com/kuchkin/mumax3-gneb/data"
)

func MyDot(a, b *data.Slice) float32 {
	N := a.Len()
	cfg := make1DConf(N)
	NN := b.Size()
	out := reduceBuf(0)
	k_mydot_async(out, a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z),
		b.DevPtr(X), b.DevPtr(Y), b.DevPtr(Z), N, NN[Z], cfg)
	return copyback(out)
}
