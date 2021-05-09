package knn

import (
	"fmt"
	"math"
	"sort"
)

type IndexDistTuple struct {
	Index int
	Dist  float64
}

type distanceFunction func([]float64, []float64) float64
type KNN struct {
	train  [][]float64
	labels []string
	k      int
	distFn distanceFunction
}

func NewKNN(k int, distFn distanceFunction) KNN {
	return KNN{
		k:      k,
		distFn: distFn,
	}
}

func (knn *KNN) Fit(x [][]float64, y []string) error {
	if len(x) != len(y) {
		return fmt.Errorf("x length of (%v) must equal y length (%v)", len(x), len(y))
	}
	knn.train = x
	knn.labels = y

	return nil
}

func (knn *KNN) Predict(x [][]float64) []string {
	// check that training data exists
	predictions := []string{}
	for _, row := range x {

		class := knn.PredictOne(row)
		predictions = append(predictions, class)
	}

	return predictions
}

func (knn *KNN) PredictOne(x []float64) string {
	f := []IndexDistTuple{}

	for i, v := range knn.train {
		dist := knn.distFn(x, v)
		f = append(f, IndexDistTuple{
			Index: i,
			Dist:  dist,
		})
	}

	sort.Slice(f, func(i, j int) bool { return f[i].Dist < f[j].Dist })

	classCount := make(map[string]int)
	for j := 0; j < knn.k; j++ {
		idt := f[j]
		class := knn.labels[idt.Index]
		classCount[class]++
	}
	return maxClassCount(classCount)
}

func maxClassCount(classCount map[string]int) string {
	maxClass := ""
	maxValue := -1
	for k, v := range classCount {
		if v > maxValue {
			maxValue = v
			maxClass = k
		}
	}

	return maxClass
}

func EuclideanDistance(a []float64, b []float64) float64 {
	if len(a) != len(b) {
		return -1
	}

	sum := float64(0)
	//sqrt of the sum of x1 - x2 squared
	for i, v := range a {
		sum += math.Pow(v-b[i], 2)
	}

	return math.Sqrt(sum)
}
