package cuda

import (
	"unsafe"

	"github.com/kuchkin/mumax3-gneb/data"
	"github.com/kuchkin/mumax3-gneb/util"
)

// Add effective field due to bulk Dzyaloshinskii-Moriya interaction to Beff.
// See dmibulk.cu
func AddDMIBulk(Beff *data.Slice, m *data.Slice, Aex_red, D_red SymmLUT, Msat MSlice, regions *Bytes, mesh *data.Mesh, OpenBC bool) {
	cellsize := mesh.CellSize()
	N := Beff.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)
	noi := mesh.NumberOfImages()
	gneb := mesh.GNEB_code()
	var openBC byte
	if OpenBC {
		openBC = 1
	}
	// var gneb2D byte
	// if GNEB2D {
	// 	gneb2D = 1
	// 	noi = N[Z]
	// }
	// var gneb3D byte
	// if (GNEB3D && !GNEB2D) {
	// 	gneb3D = 1
	// }
	if gneb == 1 || gneb == 2 {
		k_gneb_adddmibulk_async(Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
			m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
			Msat.DevPtr(0), Msat.Mul(0),
			unsafe.Pointer(Aex_red), unsafe.Pointer(D_red), regions.Ptr,
			float32(cellsize[X]), float32(cellsize[Y]), float32(cellsize[Z]), N[X], N[Y], N[Z], noi, mesh.PBC_code(), openBC, gneb, cfg)
	} else {
		k_adddmibulk_async(Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
			m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
			Msat.DevPtr(0), Msat.Mul(0),
			unsafe.Pointer(Aex_red), unsafe.Pointer(D_red), regions.Ptr,
			float32(cellsize[X]), float32(cellsize[Y]), float32(cellsize[Z]), N[X], N[Y], N[Z], mesh.PBC_code(), openBC, cfg)
	}

}
