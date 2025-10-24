import numpy as np
import pandas as pd
import math
from pathlib import Path

C = 299_792_458.0

def gauss_newton_tdoa_multistart(gw_coords, t_ns, starts, max_iter=100, tol=1e-9):
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
            ranges = np.sqrt((xx - gw_coords[:,0])**2 + (yy - gw_coords[:,1])**2)
            pred = tt0 + ranges / C
            residuals = pred - t_meas
            J = np.zeros((3,3))
            for i in range(3):
                ri = ranges[i]
                if ri == 0:
                    J[i,0] = J[i,1] = 0.0
                else:
                    J[i,0] = (xx - gw_coords[i,0]) / (ri * C)
                    J[i,1] = (yy - gw_coords[i,1]) / (ri * C)
                J[i,2] = 1.0
            try:
                delta, *_ = np.linalg.lstsq(J, -residuals, rcond=None)
            except np.linalg.LinAlgError:
                break
            xx += float(delta[0]); yy += float(delta[1]); tt0 += float(delta[2])
            if np.linalg.norm(delta) < tol:
                resnorm = np.linalg.norm(residuals)
                if resnorm < best_sol[5]:
                    best_sol = (xx, yy, tt0, True, k+1, resnorm)
                break
        else:
            ranges = np.sqrt((xx - gw_coords[:,0])**2 + (yy - gw_coords[:,1])**2)
            pred = tt0 + ranges / C
            residuals = pred - t_meas
            resnorm = np.linalg.norm(residuals)
            if resnorm < best_sol[5]:
                best_sol = (xx, yy, tt0, False, max_iter, resnorm)
    return best_sol

def simulate_lorawan_tdoa(n_samples=100, noise_std_ns=5.0, area_x=(-200,1200), area_y=(-500,1400), random_seed=31415):
    rng = np.random.default_rng(random_seed)
    gw_coords = np.array([[0.0,0.0],[1000.0,0.0],[500.0,866.0254]])
    rows = []
    for s in range(n_samples):
        true_x = float(rng.uniform(*area_x))
        true_y = float(rng.uniform(*area_y))
        t0 = rng.uniform(0,1.0)
        dists = np.sqrt((gw_coords[:,0]-true_x)**2 + (gw_coords[:,1]-true_y)**2)
        arrival_times_s = t0 + dists / C
        arrival_times_ns = np.round(arrival_times_s*1e9).astype(np.int64)
        noisy_ns = arrival_times_ns + rng.normal(0, noise_std_ns, size=3).astype(np.int64)
        starts = [
            (gw_coords[:,0].mean(), gw_coords[:,1].mean()),
            (gw_coords[0,0], gw_coords[0,1]),
            (gw_coords[1,0], gw_coords[1,1]),
            (gw_coords[2,0], gw_coords[2,1])
        ]
        sol_x, sol_y, sol_t0, converged, iters, resnorm = gauss_newton_tdoa_multistart(gw_coords, noisy_ns, starts)
        resnorm_threshold = 1e-7
        pos_err_threshold = 5000.0
        if sol_x is None or resnorm > resnorm_threshold:
            rec_x = rec_y = np.nan; converged_flag = False; pos_err = np.nan
        else:
            rec_x = sol_x; rec_y = sol_y; converged_flag = converged
            pos_err = math.hypot(rec_x - true_x, rec_y - true_y)
            if pos_err > pos_err_threshold:
                rec_x = rec_y = np.nan; converged_flag = False; pos_err = np.nan
        rows.append({
            "sample_id": s+1, "true_x_m": true_x, "true_y_m": true_y,
            "t_gw1_ns": int(noisy_ns[0]), "t_gw2_ns": int(noisy_ns[1]), "t_gw3_ns": int(noisy_ns[2]),
            "recov_x_m": rec_x, "recov_y_m": rec_y, "pos_error_m": pos_err,
            "solver_converged": bool(converged_flag), "solver_iters": int(iters), "solver_resnorm_s": float(resnorm),
            "gw1_x_m": gw_coords[0,0], "gw1_y_m": gw_coords[0,1],
            "gw2_x_m": gw_coords[1,0], "gw2_y_m": gw_coords[1,1],
            "gw3_x_m": gw_coords[2,0], "gw3_y_m": gw_coords[2,1]
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = simulate_lorawan_tdoa()
    out_path = Path("lorawan_tdoa_samples.csv")
    df.to_csv(out_path, index=False)
    valid = df['pos_error_m'].dropna()
    avg_err = valid.mean() if len(valid)>0 else float('nan')
    med_err = valid.median() if len(valid)>0 else float('nan')
    print(f"Saved CSV to: {out_path.resolve()}")
    print(f"Valid solves: {df['solver_converged'].sum()} / {len(df)}")
    print(f"Average position error: {avg_err:.3f} m")
    print(f"Median position error: {med_err:.3f} m")
