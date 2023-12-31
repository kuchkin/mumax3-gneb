package main

import (
	"log"
	"strconv"
	"strings"

	"github.com/kuchkin/mumax3-gneb/data"
	"github.com/kuchkin/mumax3-gneb/util"
)

func resize(f *data.Slice, arg string) {
	s := parseSize(arg)
	resized := data.Resample(f, s)
	*f = *resized
}

func parseSize(arg string) (size [3]int) {
	words := strings.Split(arg, "x")
	if len(words) != 3 {
		log.Fatal("resize: need N0xN1xN2 argument")
	}
	for i, w := range words {
		v, err := strconv.Atoi(w)
		util.FatalErr(err)
		size[i] = v
	}
	return
}
