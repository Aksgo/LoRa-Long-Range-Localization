import numpy as np
import pandas as pd
import math
from pathlib import Path

# --- Constants ---
C = 299_792_458.0  # speed of light (m/s)

# --- Gauss–Newton TDOA Solver (multi-start) ---
def gauss_newton_tdoa_multistart(gw_coords, t_ns, starts, max_iter=100, tol=1e-9):
    """Estimate (x, y, t0) of node from gateway nanosecond timestamps."""
    t_meas = np.array(t_ns, dtype=float) * 1e-9
    best_sol = (None, None, None, False, 0, np.inf)

    for start in starts:
        if len(start) == 3:
            x, y, t0 = start
        else:
            x, y = start
            centroid = np.array([x, y])
            d_to_centroid = np.linalg.norm(gw_coords - centroid, axis=1).min()
            t0 = t_meas.min() - d_to_centroid / C

        xx, yy, tt0 = x, y, t0

        for k in range(max_iter):
            ranges = np.sqrt((xx - gw_coords[:, 0])**2 + (yy - gw_coords[:, 1])**2)
            pred = tt0 + ranges / C
            residuals = pred - t_meas

            J = np.zeros((3, 3))
            for i in range(3):
                ri = ranges[i]
                if ri == 0:
                    J[i, 0] = J[i, 1] = 0.0
                else:
                    J[i, 0] = (xx - gw_coords[i, 0]) / (ri * C)
                    J[i, 1] = (yy - gw_coords[i, 1]) / (ri * C)
                J[i, 2] = 1.0

            try:
                delta, *_ = np.linalg.lstsq(J, -residuals, rcond=None)
            except np.linalg.LinAlgError:
                break

            xx += float(delta[0])
            yy += float(delta[1])
            tt0 += float(delta[2])

            if np.linalg.norm(delta) < tol:
                resnorm = np.linalg.norm(residuals)
                if resnorm < best_sol[5]:
                    best_sol = (xx, yy, tt0, True, k + 1, resnorm)
                break
        else:
            ranges = np.sqrt((xx - gw_coords[:, 0])**2 + (yy - gw_coords[:, 1])**2)
            pred = tt0 + ranges / C
            residuals = pred - t_meas
            resnorm = np.linalg.norm(residuals)
            if resnorm < best_sol[5]:
                best_sol = (xx, yy, tt0, False, max_iter, resnorm)
    return best_sol

# --- Dataset Processor ---
def process_tdoa_dataset(csv_path):
    """
    Reads dataset with gateway timestamp columns (t_gw1_ns, t_gw2_ns, t_gw3_ns)
    and solves for the node location using Gauss–Newton TDOA method.
    """
    df = pd.read_csv(csv_path)

    # Extract gateway coordinates (assumed constant for all samples)
    gw_coords = np.array([
        [df["gw1_x_m"].iloc[0], df["gw1_y_m"].iloc[0]],
        [df["gw2_x_m"].iloc[0], df["gw2_y_m"].iloc[0]],
        [df["gw3_x_m"].iloc[0], df["gw3_y_m"].iloc[0]],
    ])

    results = []
    for i, row in df.iterrows():
        t_ns = [row["t_gw1_ns"], row["t_gw2_ns"], row["t_gw3_ns"]]
        starts = [
            (gw_coords[:, 0].mean(), gw_coords[:, 1].mean()),
            (gw_coords[0, 0], gw_coords[0, 1]),
            (gw_coords[1, 0], gw_coords[1, 1]),
            (gw_coords[2, 0], gw_coords[2, 1]),
        ]

        sol_x, sol_y, sol_t0, converged, iters, resnorm = gauss_newton_tdoa_multistart(
            gw_coords, t_ns, starts
        )

        if sol_x is not None and converged:
            results.append({
                "sample_id": row.get("sample_id", i + 1),
                "recov_x_m": sol_x,
                "recov_y_m": sol_y,
                "solver_converged": converged,
                "solver_iters": iters,
                "solver_resnorm_s": resnorm,
            })
        else:
            results.append({
                "sample_id": row.get("sample_id", i + 1),
                "recov_x_m": np.nan,
                "recov_y_m": np.nan,
                "solver_converged": False,
                "solver_iters": iters,
                "solver_resnorm_s": resnorm,
            })

    out_df = pd.DataFrame(results)
    out_path = Path("tdoa_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"✅ Results saved to: {out_path.resolve()}")
    print(out_df.head())
    return out_df

# --- Main Entry Point ---
if __name__ == "__main__":
    input_csv = "lorawan_data.csv"
    df_results = process_tdoa_dataset(input_csv)
