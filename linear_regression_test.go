package main

import "testing"

func TestLinearRegression(t testing.T) {
	x := [][]int{
		[]int{1, 2, 3, 4},
		[]int{4, 5, 6, 7},
	}

	y := []int{1, 2}
	linreg := NewLinearRegression()
	linreg.Fit(x, y)
	out := linreg.Predict(x)
}
