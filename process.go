package main

import (
    "fmt"
    "math"
    "math/rand"
    "os"
    "encoding/csv"
    "strconv"
    "sync"
    "time"

    "gonum.org/v1/gonum/mat"
)

const C = 299_792_458.0

type GW struct {
    X, Y float64
}

type Sample struct {
    TrueX, TrueY float64
    TGW          [3]int64
    RecX, RecY   float64
    PosErr       float64
    Converged    bool
    Iters        int
    ResNorm      float64
}

// Gauss-Newton TDOA Solver
func GaussNewtonTDOA(gwCoords [3]GW, tNs [3]int64, starts [][3]float64, maxIter int, tol float64) (float64, float64, float64, bool, int, float64) {
    tMeas := [3]float64{}
    for i := 0; i < 3; i++ {
        tMeas[i] = float64(tNs[i]) * 1e-9
    }

    bestRes := math.Inf(1)
    var bestX, bestY, bestT0 float64
    var converged bool
    var bestIters int

    for _, start := range starts {
        xx, yy, tt0 := start[0], start[1], start[2]

        for k := 0; k < maxIter; k++ {
            ranges := [3]float64{}
            pred := [3]float64{}
            residuals := [3]float64{}
            for i := 0; i < 3; i++ {
                dx := xx - gwCoords[i].X
                dy := yy - gwCoords[i].Y
                ranges[i] = math.Hypot(dx, dy)
                pred[i] = tt0 + ranges[i]/C
                residuals[i] = pred[i] - tMeas[i]
            }

            // Jacobian
            J := mat.NewDense(3, 3, nil)
            for i := 0; i < 3; i++ {
                ri := ranges[i]
                if ri == 0 {
                    J.Set(i, 0, 0)
                    J.Set(i, 1, 0)
                } else {
                    J.Set(i, 0, (xx-gwCoords[i].X)/(ri*C))
                    J.Set(i, 1, (yy-gwCoords[i].Y)/(ri*C))
                }
                J.Set(i, 2, 1)
            }

            // Solve least squares
            r := mat.NewVecDense(3, []float64{-residuals[0], -residuals[1], -residuals[2]})
            delta := mat.NewVecDense(3, nil)
            var qr mat.QR
            qr.Factorize(J)
            err := qr.SolveTo(delta, false, r)
            if err != nil {
                break
            }

            xx += delta.AtVec(0)
            yy += delta.AtVec(1)
            tt0 += delta.AtVec(2)

            normDelta := math.Hypot(math.Hypot(delta.AtVec(0), delta.AtVec(1)), delta.AtVec(2))
            if normDelta < tol {
                resnorm := math.Hypot(math.Hypot(residuals[0], residuals[1]), residuals[2])
                if resnorm < bestRes {
                    bestX, bestY, bestT0 = xx, yy, tt0
                    converged = true
                    bestIters = k + 1
                    bestRes = resnorm
                }
                break
            }
        }
    }

    if bestRes == math.Inf(1) {
        return math.NaN(), math.NaN(), math.NaN(), false, 0, math.NaN()
    }
    return bestX, bestY, bestT0, converged, bestIters, bestRes
}

// Simulate TDOA samples
func SimulateLoRaWANTDOA(nSamples int, noiseStdNs float64) []Sample {
    gwCoords := [3]GW{{0, 0}, {1000, 0}, {500, 866.0254}}
    samples := make([]Sample, nSamples)

    rng := rand.New(rand.NewSource(31415))

    var wg sync.WaitGroup
    for s := 0; s < nSamples; s++ {
        wg.Add(1)
        go func(s int) {
            defer wg.Done()
            trueX := rng.Float64()*1400 - 200
            trueY := rng.Float64()*1900 - 500
            t0 := rng.Float64()

            arrivalTimes := [3]float64{}
            tNs := [3]int64{}
            noisyNs := [3]int64{}
            for i := 0; i < 3; i++ {
                dist := math.Hypot(gwCoords[i].X-trueX, gwCoords[i].Y-trueY)
                arrivalTimes[i] = t0 + dist/C
                tNs[i] = int64(math.Round(arrivalTimes[i] * 1e9))
                noisyNs[i] = tNs[i] + int64(rng.NormFloat64()*noiseStdNs)
            }

            starts := [][3]float64{
                {500, 288.675, 0}, // centroid
                {gwCoords[0].X, gwCoords[0].Y, 0},
                {gwCoords[1].X, gwCoords[1].Y, 0},
                {gwCoords[2].X, gwCoords[2].Y, 0},
            }

            recX, recY, _, converged, iters, resnorm := GaussNewtonTDOA(gwCoords, noisyNs, starts, 100, 1e-9)
            posErr := math.Hypot(recX-trueX, recY-trueY)

            if !converged || resnorm > 1e-7 || posErr > 5000 {
                recX, recY, posErr, converged = math.NaN(), math.NaN(), math.NaN(), false
            }

            samples[s] = Sample{
                TrueX:     trueX,
                TrueY:     trueY,
                TGW:       noisyNs,
                RecX:      recX,
                RecY:      recY,
                PosErr:    posErr,
                Converged: converged,
                Iters:     iters,
                ResNorm:   resnorm,
            }
        }(s)
    }
    wg.Wait()
    return samples
}

// Write CSV
func WriteCSV(filename string, samples []Sample) {
    f, err := os.Create(filename)
    if err != nil {
        panic(err)
    }
    defer f.Close()

    w := csv.NewWriter(f)
    defer w.Flush()

    header := []string{"sample_id","true_x_m","true_y_m","t_gw1_ns","t_gw2_ns","t_gw3_ns","recov_x_m","recov_y_m","pos_error_m","solver_converged","solver_iters","solver_resnorm_s"}
    _ = w.Write(header)

    for i, s := range samples {
        _ = w.Write([]string{
            strconv.Itoa(i + 1),
            fmt.Sprintf("%.6f", s.TrueX),
            fmt.Sprintf("%.6f", s.TrueY),
            strconv.FormatInt(s.TGW[0], 10),
            strconv.FormatInt(s.TGW[1], 10),
            strconv.FormatInt(s.TGW[2], 10),
            fmt.Sprintf("%.6f", s.RecX),
            fmt.Sprintf("%.6f", s.RecY),
            fmt.Sprintf("%.6f", s.PosErr),
            strconv.FormatBool(s.Converged),
            strconv.Itoa(s.Iters),
            fmt.Sprintf("%.6f", s.ResNorm),
        })
    }
}

func main() {
    start := time.Now()
    samples := SimulateLoRaWANTDOA(100, 5.0)
    WriteCSV("lorawan_tdoa_samples_go.csv", samples)
    fmt.Println("Simulation complete in", time.Since(start))
}
