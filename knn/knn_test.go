package knn

import (
	"fmt"
	"testing"
)

func TestKNNPredict(t *testing.T) {
	sampleData := [][]float64{
		{1, 1, 1},
		{2, 2, 2},
		{3, 3, 3},

		{10, 10, 10},
		{11, 11, 11},
		{12, 12, 12},
	}

	sampleY := []string{
		"Class-A",
		"Class-A",
		"Class-A",

		"Class-B",
		"Class-B",
		"Class-B",
	}

	knn := NewKNN(2, EuclideanDistance)
	err := knn.Fit(sampleData, sampleY)
	if err != nil {
		t.Errorf(err.Error())
	}

	test_ys := knn.Predict(sampleData)
	for _, y := range test_ys {
		fmt.Println(y)
	}
}
