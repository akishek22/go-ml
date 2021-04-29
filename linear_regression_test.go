package main

import (
	"fmt"
	"testing"
)

func TestLinearRegression(t testing.T) {
	x := [][]float64{
		{1, 2, 3, 4},
		{4, 5, 6, 7},
	}

	y := []float64{1, 2}
	linreg := NewLinearRegression()
	linreg.Fit(x, y)
	out := linreg.Predict(x)

	fmt.Println(out)
}
