package main

type LinearRegressionParams struct {
}

type LinearRegressionModel struct {
	Weights   []float64
	Intercept float64
}

type LinearRegression struct {
	Params LinearRegressionParams
	Model  LinearRegressionModel
}

func NewLinearRegression() LinearRegression {
	return LinearRegression{}
}

func (lr *LinearRegression) Fit(x [][]float64, y []float64) {
	// x's squared

	// x1*y, x2*y... etc...
	// x1 * x2

}

func (lr *LinearRegression) Predict(x [][]float64) []float64 {
	return nil
}

func validate_input(x [][]float64, y []float64) error {
	// x len must equal y len

	return nil
}
