package main

import (
	"fmt"
	"testing"
)

func TestLinearRegression(t *testing.T) {
	x := [][]float64{
		{1},
		{2},

		{3},
		{4},

		{5},
		{6},
	}

	y := []float64{1, 3, 5, 7, 9, 11}
	linreg := NewLinearRegression(0.01, 1050)
	linreg.Fit(x, y)
	out := linreg.Predict(x)
	fmt.Println(linreg.Model.Weights, linreg.Model.Intercept)
	fmt.Println(out)
}
