package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
)

const C = 299792458.0 // speed of light (m/s)

// --- Helper Functions ---

func norm2(x, y float64) float64 {
	return math.Sqrt(x*x + y*y)
}

func lstsq3x3(J [3][3]float64, r [3]float64) (dx, dy, dt float64, ok bool) {
	// Solve J * delta = -r using Gaussian elimination
	A := J
	b := [3]float64{-r[0], -r[1], -r[2]}

	for i := 0; i < 3; i++ {
		// pivot
		maxRow := i
		for k := i + 1; k < 3; k++ {
			if math.Abs(A[k][i]) > math.Abs(A[maxRow][i]) {
				maxRow = k
			}
		}
		A[i], A[maxRow] = A[maxRow], A[i]
		b[i], b[maxRow] = b[maxRow], b[i]

		if math.Abs(A[i][i]) < 1e-15 {
			return 0, 0, 0, false
		}

		// eliminate
		for k := i + 1; k < 3; k++ {
			f := A[k][i] / A[i][i]
			for j := i; j < 3; j++ {
				A[k][j] -= f * A[i][j]
			}
			b[k] -= f * b[i]
		}
	}

	// back substitution
	x := [3]float64{}
	for i := 2; i >= 0; i-- {
		sum := b[i]
		for j := i + 1; j < 3; j++ {
			sum -= A[i][j] * x[j]
		}
		x[i] = sum / A[i][i]
	}
	return x[0], x[1], x[2], true
}

// --- Gauss–Newton TDOA Solver (multi-start) ---

type Solution struct {
	X, Y, T0   float64
	Converged  bool
	Iterations int
	ResNorm    float64
}

func GaussNewtonTDOAMultiStart(gwCoords [3][2]float64, t_ns [3]float64, starts [][2]float64, maxIter int, tol float64) Solution {
	tMeas := [3]float64{}
	for i := 0; i < 3; i++ {
		tMeas[i] = t_ns[i] * 1e-9
	}

	best := Solution{ResNorm: math.Inf(1)}

	for _, start := range starts {
		x, y := start[0], start[1]
		centroid := [2]float64{x, y}

		minD := math.Inf(1)
		for _, gw := range gwCoords {
			d := norm2(gw[0]-centroid[0], gw[1]-centroid[1])
			if d < minD {
				minD = d
			}
		}
		t0 := min(tMeas[0], min(tMeas[1], tMeas[2])) - minD/C

		xx, yy, tt0 := x, y, t0

		for k := 0; k < maxIter; k++ {
			var ranges [3]float64
			for i := 0; i < 3; i++ {
				ranges[i] = norm2(xx-gwCoords[i][0], yy-gwCoords[i][1])
			}

			var pred, residuals [3]float64
			for i := 0; i < 3; i++ {
				pred[i] = tt0 + ranges[i]/C
				residuals[i] = pred[i] - tMeas[i]
			}

			var J [3][3]float64
			for i := 0; i < 3; i++ {
				ri := ranges[i]
				if ri != 0 {
					J[i][0] = (xx - gwCoords[i][0]) / (ri * C)
					J[i][1] = (yy - gwCoords[i][1]) / (ri * C)
				}
				J[i][2] = 1.0
			}

			dx, dy, dt, ok := lstsq3x3(J, residuals)
			if !ok {
				break
			}

			xx += dx
			yy += dy
			tt0 += dt

			if math.Sqrt(dx*dx+dy*dy+dt*dt) < tol {
				resNorm := 0.0
				for _, r := range residuals {
					resNorm += r * r
				}
				resNorm = math.Sqrt(resNorm)
				if resNorm < best.ResNorm {
					best = Solution{xx, yy, tt0, true, k + 1, resNorm}
				}
				break
			}
		}
	}
	return best
}

// --- Dataset Processor ---

func ProcessTDOADataset(csvPath string) []map[string]interface{} {
	file, err := os.Open(csvPath)
	if err != nil {
		log.Fatalf("failed to open CSV: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("failed to read CSV: %v", err)
	}

	header := rows[0]
	data := rows[1:]

	col := func(name string) int {
		for i, h := range header {
			if h == name {
				return i
			}
		}
		log.Fatalf("missing column: %s", name)
		return -1
	}

	// Extract gateway coordinates
	gwCoords := [3][2]float64{}
	for i := 0; i < 3; i++ {
		x, _ := strconv.ParseFloat(data[0][col(fmt.Sprintf("gw%d_x_m", i+1))], 64)
		y, _ := strconv.ParseFloat(data[0][col(fmt.Sprintf("gw%d_y_m", i+1))], 64)
		gwCoords[i] = [2]float64{x, y}
	}

	results := []map[string]interface{}{}

	for i, row := range data {
		var t_ns [3]float64
		for j := 0; j < 3; j++ {
			t_ns[j], _ = strconv.ParseFloat(row[col(fmt.Sprintf("t_gw%d_ns", j+1))], 64)
		}

		starts := [][2]float64{
			{(gwCoords[0][0] + gwCoords[1][0] + gwCoords[2][0]) / 3,
				(gwCoords[0][1] + gwCoords[1][1] + gwCoords[2][1]) / 3},
			{gwCoords[0][0], gwCoords[0][1]},
			{gwCoords[1][0], gwCoords[1][1]},
			{gwCoords[2][0], gwCoords[2][1]},
		}

		sol := GaussNewtonTDOAMultiStart(gwCoords, t_ns, starts, 100, 1e-9)

		res := map[string]interface{}{
			"sample_id":         i + 1,
			"recov_x_m":         sol.X,
			"recov_y_m":         sol.Y,
			"solver_converged":  sol.Converged,
			"solver_iters":      sol.Iterations,
			"solver_resnorm_s":  sol.ResNorm,
		}
		results = append(results, res)
	}

	outPath := "tdoa_results.csv"
	f, err := os.Create(outPath)
	if err != nil {
		log.Fatalf("failed to create output CSV: %v", err)
	}
	defer f.Close()

	writer := csv.NewWriter(f)
	defer writer.Flush()

	writer.Write([]string{"sample_id", "recov_x_m", "recov_y_m", "solver_converged", "solver_iters", "solver_resnorm_s"})
	for _, r := range results {
		writer.Write([]string{
			fmt.Sprintf("%v", r["sample_id"]),
			fmt.Sprintf("%v", r["recov_x_m"]),
			fmt.Sprintf("%v", r["recov_y_m"]),
			fmt.Sprintf("%v", r["solver_converged"]),
			fmt.Sprintf("%v", r["solver_iters"]),
			fmt.Sprintf("%v", r["solver_resnorm_s"]),
		})
	}

	abs, _ := filepath.Abs(outPath)
	fmt.Printf("✅ Results saved to: %s\n", abs)
	return results
}

// --- Main Entry Point ---

func main() {
	inputCSV := "lorawan_data.csv"
	ProcessTDOADataset(inputCSV)
}

// --- Utilities ---

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
