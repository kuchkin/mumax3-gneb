package engine

// Effective field

import "github.com/kuchkin/mumax3-gneb/data"

var B_eff = NewVectorField("B_eff", "T", "Effective field", SetEffectiveField)

// Sets dst to the current effective field, in Tesla.
// This is the sum of all effective field terms,
// like demag, exchange, ...
func SetEffectiveField(dst *data.Slice) {
	SetDemagField(dst)    // set to B_demag...
	AddExchangeField(dst) // ...then add other terms
	AddAnisotropyField(dst)
	AddMagnetoelasticField(dst)
	B_ext.AddTo(dst)
	if !relaxing {
		B_therm.AddTo(dst)
	}
	AddCustomField(dst)
}

func SetEffectiveField4D(n, dst *data.Slice) {

	AddExchangeField4D(n, dst)

}
