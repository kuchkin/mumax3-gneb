package main

import (
	"encoding/json"
	"io"

	"github.com/kuchkin/mumax3-gneb/data"
)

func dumpJSON(f *data.Slice, info data.Meta, out io.Writer) {
	w := json.NewEncoder(out)
	w.Encode(f.Tensors())
}
