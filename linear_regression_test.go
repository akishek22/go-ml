package main

import "testing"

func TestLinearRegression(t testing.T) {
	x := [][]float64{
		[]float64{1, 2, 3, 4},
		[]float64{4, 5, 6, 7},
	}

	y := []float64{1, 2}
	linreg := NewLinearRegression()
	linreg.Fit(x, y)
	out := linreg.Predict(x)
}
