package oommf

import (
	"fmt"
	"github.com/kuchkin/mumax3-gneb/data"
	"io"
	"log"
	"math"
	"strings"
	"unsafe"
)

func WriteOVF2(out io.Writer, q *data.Slice, meta data.Meta, dataformat string) {
	writeOVF2Header(out, q, meta)
	writeOVF2Data(out, q, dataformat)
	hdr(out, "End", "Segment")
}

func WriteOVF3(noi int, out io.Writer, q *data.Slice, meta data.Meta, dataformat string) {
	writeOVF3Header(noi, out, q, meta)
	writeOVF3Data(noi, out, q, dataformat)
	hdr(out, "End", "Segment")
}

func writeOVF2Header(out io.Writer, q *data.Slice, meta data.Meta) {
	gridsize := q.Size()
	cellsize := meta.CellSize

	fmt.Fprintln(out, "# OOMMF OVF 2.0")
	hdr(out, "Segment count", "1")
	hdr(out, "Begin", "Segment")
	hdr(out, "Begin", "Header")

	hdr(out, "Title", meta.Name)
	hdr(out, "meshtype", "rectangular")
	hdr(out, "meshunit", "m")

	hdr(out, "xmin", 0)
	hdr(out, "ymin", 0)
	hdr(out, "zmin", 0)

	hdr(out, "xmax", cellsize[X]*float64(gridsize[X]))
	hdr(out, "ymax", cellsize[Y]*float64(gridsize[Y]))
	hdr(out, "zmax", cellsize[Z]*float64(gridsize[Z]))

	name := meta.Name
	var labels []interface{}
	if q.NComp() == 1 {
		labels = []interface{}{name}
	} else {
		for i := 0; i < q.NComp(); i++ {
			labels = append(labels, name+"_"+string('x'+i))
		}
	}
	hdr(out, "valuedim", q.NComp())
	hdr(out, "valuelabels", labels...) // TODO
	unit := meta.Unit
	if unit == "" {
		unit = "1"
	}
	if q.NComp() == 1 {
		hdr(out, "valueunits", unit)
	} else {
		hdr(out, "valueunits", unit, unit, unit)
	}

	// We don't really have stages
	//fmt.Fprintln(out, "# Desc: Stage simulation time: ", meta.TimeStep, " s") // TODO
	hdr(out, "Desc", "Total simulation time: ", meta.Time, " s")

	hdr(out, "xbase", cellsize[X]/2)
	hdr(out, "ybase", cellsize[Y]/2)
	hdr(out, "zbase", cellsize[Z]/2)
	hdr(out, "xnodes", gridsize[X])
	hdr(out, "ynodes", gridsize[Y])
	hdr(out, "znodes", gridsize[Z])
	hdr(out, "xstepsize", cellsize[X])
	hdr(out, "ystepsize", cellsize[Y])
	hdr(out, "zstepsize", cellsize[Z])
	hdr(out, "End", "Header")
}

func writeOVF3Header(noi int, out io.Writer, q *data.Slice, meta data.Meta) {
	gridsize := q.Size()
	cellsize := meta.CellSize

	fmt.Fprintln(out, "# OOMMF OVF 2.0")
	hdr(out, "Segment count", "1")
	hdr(out, "Begin", "Segment")
	hdr(out, "Begin", "Header")

	hdr(out, "Title", meta.Name)
	hdr(out, "meshtype", "rectangular")
	hdr(out, "meshunit", "m")

	hdr(out, "xmin", 0)
	hdr(out, "ymin", 0)
	hdr(out, "zmin", 0)

	hdr(out, "xmax", cellsize[X]*float64(gridsize[X]))
	hdr(out, "ymax", cellsize[Y]*float64(gridsize[Y]))
	hdr(out, "zmax", cellsize[Z]*float64(gridsize[Z]*noi))

	name := meta.Name
	var labels []interface{}
	if q.NComp() == 1 {
		labels = []interface{}{name}
	} else {
		for i := 0; i < q.NComp(); i++ {
			labels = append(labels, name+"_"+string('x'+i))
		}
	}
	hdr(out, "valuedim", q.NComp())
	hdr(out, "valuelabels", labels...) // TODO
	unit := meta.Unit
	if unit == "" {
		unit = "1"
	}
	if q.NComp() == 1 {
		hdr(out, "valueunits", unit)
	} else {
		hdr(out, "valueunits", unit, unit, unit)
	}

	// We don't really have stages
	//fmt.Fprintln(out, "# Desc: Stage simulation time: ", meta.TimeStep, " s") // TODO
	hdr(out, "Desc", "Total simulation time: ", meta.Time, " s")

	hdr(out, "xbase", cellsize[X]/2)
	hdr(out, "ybase", cellsize[Y]/2)
	hdr(out, "zbase", cellsize[Z]/2)
	hdr(out, "xnodes", gridsize[X])
	hdr(out, "ynodes", gridsize[Y])
	hdr(out, "znodes", gridsize[Z]*noi)
	hdr(out, "xstepsize", cellsize[X])
	hdr(out, "ystepsize", cellsize[Y])
	hdr(out, "zstepsize", cellsize[Z])
	hdr(out, "End", "Header")
}

func WritePathToOVF(out io.Writer, q *data.Slice, meta data.Meta, noi, image int) {
	canonicalFormat := "Binary 4"

	gridsize := q.Size()
	cellsize := meta.CellSize

	fmt.Fprintln(out, "# OOMMF OVF 2.0")
	hdr(out, "Segment count", "1")
	hdr(out, "Begin", "Segment")
	hdr(out, "Begin", "Header")

	hdr(out, "Title", meta.Name)
	hdr(out, "meshtype", "rectangular")
	hdr(out, "meshunit", "m")

	hdr(out, "xmin", 0)
	hdr(out, "ymin", 0)
	hdr(out, "zmin", 0)

	hdr(out, "xmax", cellsize[X]*float64(gridsize[X]))
	hdr(out, "ymax", cellsize[Y]*float64(gridsize[Y]))
	hdr(out, "zmax", cellsize[Z]*float64(gridsize[Z]/noi))

	name := meta.Name
	var labels []interface{}
	if q.NComp() == 1 {
		labels = []interface{}{name}
	} else {
		for i := 0; i < q.NComp(); i++ {
			labels = append(labels, name+"_"+string('x'+i))
		}
	}
	hdr(out, "valuedim", q.NComp())
	hdr(out, "valuelabels", labels...) // TODO
	unit := meta.Unit
	if unit == "" {
		unit = "1"
	}
	if q.NComp() == 1 {
		hdr(out, "valueunits", unit)
	} else {
		hdr(out, "valueunits", unit, unit, unit)
	}

	// We don't really have stages
	//fmt.Fprintln(out, "# Desc: Stage simulation time: ", meta.TimeStep, " s") // TODO
	hdr(out, "Desc", "Total simulation time: ", meta.Time, " s")

	hdr(out, "xbase", cellsize[X]/2)
	hdr(out, "ybase", cellsize[Y]/2)
	hdr(out, "zbase", cellsize[Z]/2)
	hdr(out, "xnodes", gridsize[X])
	hdr(out, "ynodes", gridsize[Y])
	hdr(out, "znodes", gridsize[Z]/noi)
	hdr(out, "xstepsize", cellsize[X])
	hdr(out, "ystepsize", cellsize[Y])
	hdr(out, "zstepsize", cellsize[Z])
	hdr(out, "End", "Header")

	hdr(out, "Begin", "Data "+canonicalFormat)

	data := q.Tensors()
	size := q.Size()

	var bytes []byte

	// OOMMF requires this number to be first to check the format
	var controlnumber float32 = OVF_CONTROL_NUMBER_4
	bytes = (*[4]byte)(unsafe.Pointer(&controlnumber))[:]
	out.Write(bytes)
	ncomp := q.NComp()

	for iz := 0; iz < size[Z]/noi; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < ncomp; c++ {
					bytes = (*[4]byte)(unsafe.Pointer(&data[c][iz+image*size[Z]/noi][iy][ix]))[:]
					out.Write(bytes)
				}
			}
		}
	}

	hdr(out, "End", "Data "+canonicalFormat)

}

func writeOVF2Data(out io.Writer, q *data.Slice, dataformat string) {
	canonicalFormat := ""
	switch strings.ToLower(dataformat) {
	case "text":
		canonicalFormat = "Text"
		hdr(out, "Begin", "Data "+canonicalFormat)
		writeOVFText(out, q)
	case "binary", "binary 4":
		canonicalFormat = "Binary 4"
		hdr(out, "Begin", "Data "+canonicalFormat)
		writeOVF2DataBinary4(out, q)
	default:
		log.Fatalf("Illegal OMF data format: %v. Options are: Text, Binary 4", dataformat)
	}
	hdr(out, "End", "Data "+canonicalFormat)
}

func writeOVF3Data(noi int, out io.Writer, q *data.Slice, dataformat string) {
	canonicalFormat := ""
	switch strings.ToLower(dataformat) {
	case "text":
		canonicalFormat = "Text"
		hdr(out, "Begin", "Data "+canonicalFormat)
		writeOVFText(out, q)
	case "binary", "binary 4":
		canonicalFormat = "Binary 4"
		hdr(out, "Begin", "Data "+canonicalFormat)
		writeOVF3DataBinary4(noi, out, q)
	default:
		log.Fatalf("Illegal OMF data format: %v. Options are: Text, Binary 4", dataformat)
	}
	hdr(out, "End", "Data "+canonicalFormat)
}

func writeOVF2DataBinary4(out io.Writer, array *data.Slice) {

	//w.count(w.out.Write((*(*[1<<31 - 1]byte)(unsafe.Pointer(&list[0])))[0 : 4*len(list)])) // (shortcut)

	data := array.Tensors()
	size := array.Size()

	var bytes []byte

	// OOMMF requires this number to be first to check the format
	var controlnumber float32 = OVF_CONTROL_NUMBER_4
	bytes = (*[4]byte)(unsafe.Pointer(&controlnumber))[:]
	out.Write(bytes)
	ncomp := array.NComp()

	for iz := 0; iz < size[Z]; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < ncomp; c++ {
					bytes = (*[4]byte)(unsafe.Pointer(&data[c][iz][iy][ix]))[:]
					out.Write(bytes)
				}
			}
		}
	}

}

func writeOVF3DataBinary4(noi int, out io.Writer, array *data.Slice) {

	//w.count(w.out.Write((*(*[1<<31 - 1]byte)(unsafe.Pointer(&list[0])))[0 : 4*len(list)])) // (shortcut)

	data := array.Tensors()
	size := array.Size()

	var bytes []byte

	// OOMMF requires this number to be first to check the format
	var controlnumber float32 = OVF_CONTROL_NUMBER_4
	bytes = (*[4]byte)(unsafe.Pointer(&controlnumber))[:]
	out.Write(bytes)
	ncomp := array.NComp()

	for ii := 0; ii < noi; ii++ {
		for iz := 0; iz < size[Z]; iz++ {
			for iy := 0; iy < size[Y]; iy++ {
				for ix := 0; ix < size[X]; ix++ {
					for c := 0; c < ncomp; c++ {
						bytes = (*[4]byte)(unsafe.Pointer(&data[c][iz][iy][ix]))[:]
						out.Write(bytes)
					}
				}
			}
		}
	}

	//

}

func readOVF2DataBinary4(in io.Reader, array *data.Slice) {
	size := array.Size()
	data := array.Tensors()

	// OOMMF requires this number to be first to check the format
	controlnumber := readFloat32(in)
	if controlnumber != OVF_CONTROL_NUMBER_4 {
		panic("invalid OVF2 control number: " + fmt.Sprint(controlnumber))
	}

	ncomp := array.NComp()
	for iz := 0; iz < size[Z]; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < ncomp; c++ {
					data[c][iz][iy][ix] = readFloat32(in)
				}
			}
		}
	}
}

// fully read buf, panic on error
func readFull(in io.Reader, buf []byte) {
	_, err := io.ReadFull(in, buf)
	if err != nil {
		panic(err)
	}
	return
}

// read float32 in machine endianess, panic on error
func readFloat32(in io.Reader) float32 {
	var bytes4 [4]byte
	bytes := bytes4[:]
	readFull(in, bytes)
	return *((*float32)(unsafe.Pointer(&bytes4)))
}

// read float64 in machine endianess, panic on error
func readFloat64(in io.Reader) float64 {
	var bytes8 [8]byte
	bytes := bytes8[:]
	readFull(in, bytes)
	return *((*float64)(unsafe.Pointer(&bytes8)))
}

func readOVF2DataBinary8(in io.Reader, array *data.Slice) {
	size := array.Size()
	data := array.Tensors()

	// OOMMF requires this number to be first to check the format
	controlnumber := readFloat64(in)
	if controlnumber != OVF_CONTROL_NUMBER_8 {
		panic("invalid OVF2 control number: " + fmt.Sprint(controlnumber))
	}

	ncomp := array.NComp()
	for iz := 0; iz < size[Z]; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < ncomp; c++ {
					data[c][iz][iy][ix] = float32(readFloat64(in))
				}
			}
		}
	}
}

func readOVFGNEB8(in1, in2 io.Reader, array *data.Slice, noi, sl int) {
	size := array.Size()
	data := array.Tensors()
	ncomp := array.NComp()

	controlnumber := readFloat64(in1)
	if controlnumber != OVF_CONTROL_NUMBER_8 {
		panic("invalid OVF2 control number: " + fmt.Sprint(controlnumber))
	}
	for iz := 0; iz < size[Z]/noi; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < ncomp; c++ {
					data[c][iz][iy][ix] = float32(readFloat64(in1))
				}
			}
		}
	}

	controlnumber = readFloat64(in2)
	if controlnumber != OVF_CONTROL_NUMBER_8 {
		panic("invalid OVF2 control number: " + fmt.Sprint(controlnumber))
	}
	for iz := 0; iz < size[Z]/noi; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < ncomp; c++ {
					data[c][iz+size[Z]*(sl-1)/noi][iy][ix] = float32(readFloat64(in2))
				}
			}
		}
	}

	//INTERPOLATION
	for iz := size[Z] / noi; iz < size[Z]*(noi-1)/noi; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {

				q := iz % (size[Z] / noi)
				n0 := data[0][q][iy][ix]
				n1 := data[1][q][iy][ix]
				n2 := data[2][q][iy][ix]
				q = q + size[Z]*(noi-1)/noi
				m0 := data[0][q][iy][ix]
				m1 := data[1][q][iy][ix]
				m2 := data[2][q][iy][ix]

				c0 := n1*m2 - n2*m1
				c1 := n2*m0 - n0*m2
				c2 := n0*m1 - n1*m0
				dc := math.Sqrt(float64(c0*c0 + c1*c1 + c2*c2))
				dab := float64(n0*m0 + n1*m1 + n2*m2)

				Th := math.Atan2(dc, dab)
				p := iz / (size[Z] / noi)
				th := Th * float64(p) / float64(noi-1.0)
				pref := float32(math.Sin(th) / math.Sin(Th))

				b0 := c1*n2 - c2*n1
				b1 := c2*n0 - c0*n2
				b2 := c0*n1 - c1*n0

				data[0][iz][iy][ix] = n0*float32(math.Cos(th)) + b0*pref
				data[1][iz][iy][ix] = n1*float32(math.Cos(th)) + b1*pref
				data[2][iz][iy][ix] = n2*float32(math.Cos(th)) + b2*pref

			}
		}
	}

}

func readOVFGNEB4(in1, in2 io.Reader, array *data.Slice, noi, sl int) {
	size := array.Size()
	data := array.Tensors()
	ncomp := array.NComp()

	controlnumber := readFloat32(in1)
	if controlnumber != OVF_CONTROL_NUMBER_4 {
		panic("invalid OVF2 control number: " + fmt.Sprint(controlnumber))
	}
	for iz := 0; iz < size[Z]/noi; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < ncomp; c++ {
					data[c][iz][iy][ix] = readFloat32(in1)
				}
			}
		}
	}

	controlnumber = readFloat32(in2)
	if controlnumber != OVF_CONTROL_NUMBER_4 {
		panic("invalid OVF2 control number: " + fmt.Sprint(controlnumber))
	}
	for iz := 0; iz < size[Z]/noi; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < ncomp; c++ {
					data[c][iz+size[Z]*(sl-1)/noi][iy][ix] = readFloat32(in2)
				}
			}
		}
	}

	//INTERPOLATION
	for iz := size[Z] / noi; iz < size[Z]*(noi-1)/noi; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {

				q := iz % (size[Z] / noi)
				n0 := data[0][q][iy][ix]
				n1 := data[1][q][iy][ix]
				n2 := data[2][q][iy][ix]
				q = q + size[Z]*(noi-1)/noi
				m0 := data[0][q][iy][ix]
				m1 := data[1][q][iy][ix]
				m2 := data[2][q][iy][ix]

				// c0 := n1*m2 - n2*m1
				// c1 := n2*m0 - n0*m2
				// c2 := n0*m1 - n1*m0
				// dc := math.Sqrt(float64(c0*c0 + c1*c1 + c2*c2))
				// dab := float64(n0*m0 + n1*m1 + n2*m2)

				// Th := math.Atan2(dc, dab)
				// p := iz / (size[Z] / noi)
				// th := Th * float64(p) / float64(noi-1.0)
				// pref := float32(math.Sin(th) / math.Sin(Th))
				// if dc < 1.0e-5 {
				// 	Th = 0.0
				// 	pref = 0.0
				// }

				// b0 := c1*n2 - c2*n1
				// b1 := c2*n0 - c0*n2
				// b2 := c0*n1 - c1*n0

				// data[0][iz][iy][ix] = n0*float32(math.Cos(th)) + b0*pref
				// data[1][iz][iy][ix] = n1*float32(math.Cos(th)) + b1*pref
				// data[2][iz][iy][ix] = n2*float32(math.Cos(th)) + b2*pref

				pref := float32(iz / (size[Z] / noi)) / float32(noi-1.0);

				data[0][iz][iy][ix] = n0*(1.0-pref) + m0*pref
				data[1][iz][iy][ix] = n1*(1.0-pref) + m1*pref
				data[2][iz][iy][ix] = n2*(1.0-pref) + m2*pref


			}
		}
	}

}
