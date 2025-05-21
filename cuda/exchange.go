package cuda

import (
	"unsafe"

	"github.com/kuchkin/mumax3-gneb/data"
)

// Add exchange field to Beff.
// 	m: normalized magnetization
// 	B: effective field in Tesla
// 	Aex_red: Aex / (Msat * 1e18 m2)
// see exchange.cu
func AddExchange(B, m *data.Slice, Aex_red SymmLUT, Msat MSlice, regions *Bytes, mesh *data.Mesh, JZ float32) {
	c := mesh.CellSize()
	wx := float32(2 / (c[X] * c[X]))
	wy := float32(2 / (c[Y] * c[Y]))
	wz := float32(2 / (c[Z] * c[Z]))
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	noi := mesh.NumberOfImages()
	gneb := mesh.GNEB_code()

	if gneb == 1 || gneb == 2 {
		k_gneb_addexchange_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
			m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
			Msat.DevPtr(0), Msat.Mul(0),
			unsafe.Pointer(Aex_red), regions.Ptr,
			wx, wy, wz, N[X], N[Y], N[Z], noi, pbc, gneb, JZ, cfg)
	} else {
		k_addexchange_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
			m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
			Msat.DevPtr(0), Msat.Mul(0),
			unsafe.Pointer(Aex_red), regions.Ptr,
			wx, wy, wz, N[X], N[Y], N[Z], pbc, JZ, cfg)
	}
}

func AddExchange4D(n, B *data.Slice, Aex_red SymmLUT, kappa float32, Msat MSlice, regions *Bytes, mesh *data.Mesh) {
	// c := mesh.CellSize()
	// wx := float32(2 / (c[X] * c[X]))
	// wy := float32(2 / (c[Y] * c[Y]))
	// wz := float32(2 / (c[Z] * c[Z]))
	// N := mesh.Size()
	// pbc := mesh.PBC_code()
	// cfg := make3DConf(N)
	// k_addexchange4D_async(n.DevPtr(X), B.DevPtr(X),
	// 	Msat.DevPtr(0), Msat.Mul(0), kappa,
	// 	unsafe.Pointer(Aex_red), regions.Ptr,
	// 	wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)



	c := mesh.CellSize()
	wx := float32(2 / (c[X] * c[X]))
	wy := float32(2 / (c[Y] * c[Y]))
	wz := float32(2 / (c[Z] * c[Z]))
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	noi := mesh.NumberOfImages()
	gneb := mesh.GNEB_code()

	if gneb == 1 || gneb == 2 {
		k_gneb_addexchange4D_async(n.DevPtr(X), B.DevPtr(X),
		Msat.DevPtr(0), Msat.Mul(0), kappa,
		unsafe.Pointer(Aex_red), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], noi, pbc, gneb, cfg)

	} else {
		k_addexchange4D_async(n.DevPtr(X), B.DevPtr(X),
		Msat.DevPtr(0), Msat.Mul(0), kappa,
		unsafe.Pointer(Aex_red), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)
	}

	

}

// Finds the average exchange strength around each cell, for debugging.
func ExchangeDecode(dst *data.Slice, Aex_red SymmLUT, regions *Bytes, mesh *data.Mesh) {
	c := mesh.CellSize()
	wx := float32(2 / (c[X] * c[X]))
	wy := float32(2 / (c[Y] * c[Y]))
	wz := float32(2 / (c[Z] * c[Z]))
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	k_exchangedecode_async(dst.DevPtr(0), unsafe.Pointer(Aex_red), regions.Ptr, wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)
}
