package main

import (
	"fmt"
	"math"
)

type LinearRegressionParams struct {
	LearningRate float64
	N_Epochs     int
}

type LinearRegressionModel struct {
	Weights   []float64
	Intercept float64
}

type LinearRegression struct {
	Params LinearRegressionParams
	Model  LinearRegressionModel
}

func NewLinearRegression(learning_rate float64, n_epochs int) LinearRegression {
	return LinearRegression{
		Params: LinearRegressionParams{
			LearningRate: learning_rate,
			N_Epochs:     n_epochs,
		},

		Model: LinearRegressionModel{
			Weights:   []float64{},
			Intercept: 0,
		},
	}
}

func (lr *LinearRegression) Fit(x [][]float64, y []float64) {
	lr.stochasticGradientDescent(x, y, lr.Params.LearningRate, lr.Params.N_Epochs)
}

func (lr *LinearRegression) Predict(x [][]float64) []float64 {
	predictions := []float64{}

	for _, row := range x {
		predictions = append(predictions, lr.PredictOne(row))
	}
	return predictions
}

func (lr *LinearRegression) PredictOne(x []float64) float64 {
	// y = b_1x + b_0
	yhat := float64(0)
	for i, b := range lr.Model.Weights {
		yhat += x[i] * b
	}

	yhat += lr.Model.Intercept

	return yhat
}

func (lr *LinearRegression) stochasticGradientDescent(train [][]float64, y []float64, learning_rate float64, n_epochs int) {
	lr.Model.Weights = make([]float64, len(train[0]))

	for epoch := 0; epoch < n_epochs; epoch++ {
		sum_error := float64(0)
		for i, row := range train {
			result := lr.PredictOne(row)
			err := result - y[i]
			sum_error += math.Pow(err, 2)
			lr.Model.Intercept = lr.Model.Intercept - learning_rate*err
			for j := range lr.Model.Weights {
				lr.Model.Weights[j] = lr.Model.Weights[j] - learning_rate*err*row[j]
			}
		}
		fmt.Printf("\n**Epoch = %d, Learning Rate = %f, Error = %2f\n", epoch, learning_rate, sum_error)
	}
}
