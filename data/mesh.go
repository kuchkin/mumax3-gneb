package data

import (
	"fmt"
	"log"
)

// Mesh stores info of a finite-difference mesh.
type Mesh struct {
	gridSize [3]int
	cellSize [3]float64
	pbc      [3]int
	Unit     string // unit of cellSize, default: "m"
	noi      int
	gneb     [2]int
}

// Images stores info of a number of images.
// type Images struct {
// 	noi int
// }

// Retruns a new mesh with N0 x N1 x N2 cells of size cellx x celly x cellz.
// Optional periodic boundary conditions (pbc): number of repetitions
// in X, Y, Z direction. 0,0,0 means no periodicity.
func NewMesh(N0, N1, N2 int, cellx, celly, cellz float64, pbc ...int) *Mesh {
	var pbc3 [3]int
	if len(pbc) == 3 {
		copy(pbc3[:], pbc)
	} else {
		if len(pbc) != 0 {
			log.Panic("mesh: need 0 or 3 PBC arguments, got:", pbc)
		}
	}
	size := [3]int{N0, N1, N2}
	gneb := [2]int{0, 0}
	return &Mesh{size, [3]float64{cellx, celly, cellz}, pbc3, "m", 1, gneb}
}




func NewMeshGneb(N0, N1, N2, noi, gneb2D, gneb3D int, cellx, celly, cellz float64, pbc ...int) *Mesh {
	var pbc3 [3]int
	if len(pbc) == 3 {
		copy(pbc3[:], pbc)
	} else {
		if len(pbc) != 0 {
			log.Panic("mesh: need 0 or 3 PBC arguments, got:", pbc)
		}
	}
	size := [3]int{N0, N1, N2}
	gneb := [2]int{gneb2D, gneb3D}
	return &Mesh{size, [3]float64{cellx, celly, cellz}, pbc3, "m", noi, gneb}
}

// Returns N0, N1, N2, as passed to constructor.
func (m *Mesh) Size() [3]int {
	if m == nil {
		return [3]int{0, 0, 0}
	} else {
		return m.gridSize
	}
}

// // Returns Number of Images as passed to constructor.
func (m *Mesh) NumberOfImages() int {
	return m.noi
}

// Returns cellx, celly, cellz, as passed to constructor.
func (m *Mesh) CellSize() [3]float64 {
	return m.cellSize
}

// Returns pbc (periodic boundary conditions), as passed to constructor.
func (m *Mesh) PBC() [3]int {
	return m.pbc
}
func (m *Mesh) GNEB() [2]int {
	return m.gneb
}
func (m *Mesh) SetGNEB(gneb2D, gneb3D int) {
	m.gneb = [2]int{gneb2D, gneb3D}
}

func (m *Mesh) SetPBC(nx, ny, nz int) {
	m.pbc = [3]int{nx, ny, nz}
}

// Total number of cells, not taking into account PBCs.
// 	N0 * N1 * N2
func (m *Mesh) NCell() int {
	return m.gridSize[0] * m.gridSize[1] * m.gridSize[2]
}

// WorldSize equals (grid)Size x CellSize.
func (m *Mesh) WorldSize() [3]float64 {
	return [3]float64{float64(m.gridSize[0]) * m.cellSize[0], float64(m.gridSize[1]) * m.cellSize[1], float64(m.gridSize[2]) * m.cellSize[2]}
}

// 3 bools, packed in one byte, indicating whether there are periodic boundary conditions in
// X (LSB), Y(LSB<<1), Z(LSB<<2)
func (m *Mesh) PBC_code() byte {
	var code byte
	if m.pbc[X] != 0 {
		code = 1
	}
	if m.pbc[Y] != 0 {
		code |= 2
	}
	if m.pbc[Z] != 0 {
		code |= 4
	}
	return code
}

func (m *Mesh) GNEB_code() byte {
	var code byte
	if m.gneb[X] != 0 {
		code = 1
	}
	if m.gneb[Y] != 0 {
		code |= 2
	}
	return code
}

func (m *Mesh) String() string {
	s := m.gridSize
	c := m.cellSize
	pbc := ""
	if m.pbc != [3]int{0, 0, 0} {
		pbc = fmt.Sprintf(", PBC: [%v x %v x %v],", m.pbc[0], m.pbc[1], m.pbc[2])
	}
	return fmt.Sprintf("[%v x %v x %v] x [%vm x %vm x %vm]%v", s[0], s[1], s[2], float32(c[0]), float32(c[1]), float32(c[2]), pbc)
}

// product of elements.
func prod(size [3]int) int {
	return size[0] * size[1] * size[2]
}
