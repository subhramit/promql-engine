// Copyright (c) The Thanos Community Authors.
// Licensed under the Apache License 2.0.

package function

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/prometheus/prometheus/model/labels"
	"github.com/thanos-io/promql-engine/execution/model"
	"github.com/thanos-io/promql-engine/execution/telemetry"
	"github.com/thanos-io/promql-engine/logicalplan"
	"github.com/thanos-io/promql-engine/query"
)

type holtWintersOperator struct {
	telemetry.OperatorTelemetry
	once   sync.Once
	series []labels.Labels

	pool     *model.VectorPool
	funcArgs logicalplan.Nodes
	vectorOp model.VectorOperator

	// parameters for holt winters calculation
	sf float64 // smoothing factor
	tf float64 // trend factor
}

func newHoltWintersOperator(
	pool *model.VectorPool,
	funcArgs logicalplan.Nodes,
	vectorOp model.VectorOperator,
	sf float64,
	tf float64,
	opts *query.Options,
) *holtWintersOperator {
	oper := &holtWintersOperator{
		pool:     pool,
		funcArgs: funcArgs,
		vectorOp: vectorOp,
		sf:       sf,
		tf:       tf,
	}
	oper.OperatorTelemetry = telemetry.NewTelemetry(oper, opts)
	return oper
}

func (o *holtWintersOperator) String() string {
	return fmt.Sprintf("[holt_winters](%v)", o.funcArgs)
}

func (o *holtWintersOperator) Explain() []model.VectorOperator {
	return []model.VectorOperator{o.vectorOp}
}

func (o *holtWintersOperator) Series(ctx context.Context) ([]labels.Labels, error) {
	start := time.Now()
	defer func() { o.AddExecutionTimeTaken(time.Since(start)) }()

	var err error
	o.once.Do(func() { err = o.loadSeries(ctx) })
	if err != nil {
		return nil, err
	}
	return o.series, nil
}

func (o *holtWintersOperator) GetPool() *model.VectorPool {
	return o.pool
}

func (o *holtWintersOperator) loadSeries(ctx context.Context) error {
	series, err := o.vectorOp.Series(ctx)
	if err != nil {
		return err
	}
	o.series = series
	o.pool.SetStepSize(len(series))
	return nil
}

func (o *holtWintersOperator) processVectors(vectors []model.StepVector) ([]model.StepVector, error) {
	out := o.pool.GetVectorBatch()

	for _, vector := range vectors {
		step := o.pool.GetStepVector(vector.T)

		// Use Holt-Winters calculation on each vector
		smoothedSeries, err := holtWintersCalculateSeries(vector.Samples, o.sf, o.tf)
		if err != nil {
			return nil, err
		}

		for i, value := range smoothedSeries {
			step.AppendSample(o.pool, vector.SampleIDs[i], value)
		}

		out = append(out, step)
		o.vectorOp.GetPool().PutStepVector(vector)
	}

	o.vectorOp.GetPool().PutVectors(vectors)
	return out, nil
}

// holtWintersCalculateSeries applies the Holt-Winters triple exponential smoothing (handles the actual time series calculation)
// It computes level, trend, and optionally seasonality to smooth the data.
// Parameters:
//   - series: a slice of float64 that represents the time series data.
//   - alpha, beta, gamma: smoothing factors for level, trend, and seasonality.
//   - period: the seasonality period (set to 0 for non-seasonal data).
//
// Returns:
//   - Smoothed series as a []float64.
func holtWintersCalculateSeries(series []float64, sf, tf float64) ([]float64, error) {
	if len(series) < 2 {
		return nil, fmt.Errorf("not enough data points")
	}

	smoothed := make([]float64, len(series))
	level := series[0]
	trend := series[1] - series[0]

	// First value is just the initial level
	smoothed[0] = level

	// Calculate smoothed values
	for t := 1; t < len(series); t++ {
		prevLevel := level
		value := series[t]

		if math.IsNaN(value) {
			smoothed[t] = math.NaN()
			continue
		}

		// Update level and trend
		level = sf*value + (1-sf)*(prevLevel+trend)
		trend = tf*(level-prevLevel) + (1-tf)*trend

		// Forecast for this step
		smoothed[t] = level + trend
	}

	return smoothed, nil
}
