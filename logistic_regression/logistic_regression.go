package logistic_regression

import "math"

type Params struct {
	LearningRate float64
}

type Model struct {
	Weights   []float64
	Intercept float64
}

type LogisticRegression struct {
}

func (lr *LogisticRegression) Fit(x [][]float64, y []float64) error {

	return nil
}

func (lr *LogisticRegression) PredictOne(x []float64) float64 {
	yhat = 
	return 0
}

func (lr *LogisticRegression) Predict(x [][]float64) []float64 {
	predictions := []float64{}
	for _, row := range x {
		predictions = append(predictions, lr.PredictOne(row))
	}

	return predictions
}
func Sigmod(x float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -x))
}
