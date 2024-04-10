package engine

import (
	"github.com/kuchkin/mumax3-gneb/cuda"
	"github.com/kuchkin/mumax3-gneb/data"
)

var globalmesh_ data.Mesh // mesh for m and everything that has the same size

func init() {
	DeclFunc("SetGridSize", SetGridSize, `Sets the number of cells for X,Y,Z`)
	DeclFunc("SetCellSize", SetCellSize, `Sets the X,Y,Z cell size in meters`)
	DeclFunc("SetGNEB", SetGNEB, `Sets the dimensions of images`)
	DeclFunc("SetImagesNumber", SetImagesNumber, `Sets the number of images `)
	DeclFunc("SetMesh", SetMesh, `Sets GridSize, CellSize and PBC at the same time`)
	DeclFunc("SetMeshGNEB", SetMeshGNEB, `Sets GridSize, CellSize, PBC and Gneb parameters at the same time`)
	DeclFunc("SetPBC", SetPBC, "Sets the number of repetitions in X,Y,Z to create periodic boundary "+
		"conditions. The number of repetitions determines the cutoff range for the demagnetization.")

}

func Mesh() *data.Mesh {
	checkMesh()
	return &globalmesh_
}

func arg(msg string, test bool) {
	if !test {
		panic(UserErr(msg + ": illegal arugment"))
	}
}

// Set the simulation mesh to Nx x Ny x Nz cells of given size.
// Can be set only once at the beginning of the simulation.
// TODO: dedup arguments from globals
func SetMesh(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64, pbcx, pbcy, pbcz int) {
	SetBusy(true)
	defer SetBusy(false)

	arg("GridSize", Nx > 0 && Ny > 0 && Nz > 0)
	arg("CellSize", cellSizeX > 0 && cellSizeY > 0 && cellSizeZ > 0)
	arg("PBC", pbcx >= 0 && pbcy >= 0 && pbcz >= 0)

	prevSize := globalmesh_.Size()
	pbc := []int{pbcx, pbcy, pbcz}

	if globalmesh_.Size() == [3]int{0, 0, 0} {
		// first time mesh is set
		globalmesh_ = *data.NewMesh(Nx, Ny, Nz, cellSizeX, cellSizeY, cellSizeZ, pbc...)
		M.alloc()
		regions.alloc()
	} else {
		// here be dragons
		LogOut("resizing...")

		// free everything
		conv_.Free()
		conv_ = nil
		mfmconv_.Free()
		mfmconv_ = nil
		cuda.FreeBuffers()

		// resize everything
		globalmesh_ = *data.NewMesh(Nx, Ny, Nz, cellSizeX, cellSizeY, cellSizeZ, pbc...)
		M.resize()
		regions.resize()
		geometry.buffer.Free()
		geometry.buffer = data.NilSlice(1, Mesh().Size())
		geometry.setGeom(geometry.shape)

		// remove excitation extra terms if they don't fit anymore
		// up to the user to add them again
		if Mesh().Size() != prevSize {
			B_ext.RemoveExtraTerms()
			J.RemoveExtraTerms()
		}

		if Mesh().Size() != prevSize {
			B_therm.noise.Free()
			B_therm.noise = nil
		}
	}
	lazy_gridsize = []int{Nx, Ny, Nz}
	lazy_cellsize = []float64{cellSizeX, cellSizeY, cellSizeZ}
	lazy_pbc = []int{pbcx, pbcy, pbcz}
}

func SetMeshGNEB(Nx, Ny, Nz, noi int, cellSizeX, cellSizeY, cellSizeZ float64, pbcx, pbcy, pbcz, gneb2D, gneb3D int) {
	SetBusy(true)
	defer SetBusy(false)

	arg("GridSize", Nx > 0 && Ny > 0 && Nz > 0)
	arg("CellSize", cellSizeX > 0 && cellSizeY > 0 && cellSizeZ > 0)
	arg("PBC", pbcx >= 0 && pbcy >= 0 && pbcz >= 0)
	arg("GNEB", gneb2D >= 0 && gneb3D >= 0)

	prevSize := globalmesh_.Size()
	pbc := []int{pbcx, pbcy, pbcz}
	// gneb := []int{gneb2D, gneb3D}

	if globalmesh_.Size() == [3]int{0, 0, 0} {
		// first time mesh is set
		globalmesh_ = *data.NewMeshGneb(Nx, Ny, Nz, noi, gneb2D, gneb3D, cellSizeX, cellSizeY, cellSizeZ, pbc...)
		M.alloc()
		regions.alloc()
	} else {
		// here be dragons
		LogOut("resizing...")

		// free everything
		conv_.Free()
		conv_ = nil
		mfmconv_.Free()
		mfmconv_ = nil
		cuda.FreeBuffers()

		// resize everything
		globalmesh_ = *data.NewMeshGneb(Nx, Ny, Nz, noi, gneb2D, gneb3D, cellSizeX, cellSizeY, cellSizeZ, pbc...)
		M.resize()
		regions.resize()
		geometry.buffer.Free()
		geometry.buffer = data.NilSlice(1, Mesh().Size())
		geometry.setGeom(geometry.shape)

		// remove excitation extra terms if they don't fit anymore
		// up to the user to add them again
		if Mesh().Size() != prevSize {
			B_ext.RemoveExtraTerms()
			J.RemoveExtraTerms()
		}

		if Mesh().Size() != prevSize {
			B_therm.noise.Free()
			B_therm.noise = nil
		}
	}
	lazy_gridsize = []int{Nx, Ny, Nz}
	lazy_cellsize = []float64{cellSizeX, cellSizeY, cellSizeZ}
	lazy_pbc = []int{pbcx, pbcy, pbcz}
	lazy_gneb = []int{gneb2D, gneb3D}
}

func printf(f float64) float32 {
	return float32(f)
}

// for lazy setmesh: set gridsize and cellsize in separate calls
var (
	lazy_gridsize []int
	lazy_cellsize []float64
	lazy_pbc      = []int{0, 0, 0}
	lazy_imsize   = []int{1}
	lazy_gneb     = []int{0, 0}
)

func SetGridSize(Nx, Ny, Nz int) {
	lazy_gridsize = []int{Nx, Ny, Nz}
	if lazy_cellsize != nil {
		SetMesh(Nx, Ny, Nz, lazy_cellsize[X], lazy_cellsize[Y], lazy_cellsize[Z], lazy_pbc[X], lazy_pbc[Y], lazy_pbc[Z])
	}
}

func SetCellSize(cx, cy, cz float64) {
	lazy_cellsize = []float64{cx, cy, cz}
	if lazy_gridsize != nil {
		SetMesh(lazy_gridsize[X], lazy_gridsize[Y], lazy_gridsize[Z],  cx, cy, cz, lazy_pbc[X], lazy_pbc[Y], lazy_pbc[Z])
	}
}

func SetPBC(nx, ny, nz int) {
	lazy_pbc = []int{nx, ny, nz}
	if lazy_gridsize != nil && lazy_cellsize != nil {
		SetMesh(lazy_gridsize[X], lazy_gridsize[Y], lazy_gridsize[Z],
			lazy_cellsize[X], lazy_cellsize[Y], lazy_cellsize[Z],
			lazy_pbc[X], lazy_pbc[Y], lazy_pbc[Z])
	}
}

func SetGNEB(gneb2D, gneb3D int) {
	lazy_gneb = []int{gneb2D, gneb3D}
	if lazy_gridsize != nil && lazy_cellsize != nil {
		SetMeshGNEB(lazy_gridsize[X], lazy_gridsize[Y], lazy_gridsize[Z], lazy_imsize[0],
			lazy_cellsize[X], lazy_cellsize[Y], lazy_cellsize[Z],
			lazy_pbc[X], lazy_pbc[Y], lazy_pbc[Z], lazy_gneb[X], lazy_gneb[Y])
	}
}

func SetImagesNumber(Noi int) {
	lazy_imsize = []int{Noi}
	if lazy_gridsize != nil && lazy_cellsize != nil {
		SetMeshGNEB(lazy_gridsize[X], lazy_gridsize[Y], lazy_gridsize[Z], lazy_imsize[0],
			lazy_cellsize[X], lazy_cellsize[Y], lazy_cellsize[Z],
			lazy_pbc[X], lazy_pbc[Y], lazy_pbc[Z], lazy_gneb[X], lazy_gneb[Y])
	}
}

// check if mesh is set
func checkMesh() {
	if globalmesh_.Size() == [3]int{0, 0, 0} {
		panic("need to set mesh first")
	}
	// if(lazy_imsize[0]>globalmesh_.Size()[2]){
	// 	panic("Number of images can not be bigger than Nz")
	// }
	// if((globalmesh_.Size()[2])%lazy_imsize[0] != 0){
	// 	panic("Size of each image has to be an integer value. Check Nz/NumberOfImages")
	// }
}
