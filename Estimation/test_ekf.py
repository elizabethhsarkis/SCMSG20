import numpy as np
from ekf import EKF, wrap, WHEEL_RADIUS, WHEEL_BASE

#synthetic test: no MuJoCo, truth comes from motion model
MARKER = np.array([3.325, 3.399])
#ArUco marker in the ward scene
K_ENC = 1e-4
#encoder noise
R_RB = np.diag([0.02**2, np.deg2rad(2)**2]) #2cm and 2 deg
T = 0.05 
N_STEPS = 2000

def simulate(hide_marker=False, seed=0):
    """
    Drive a wiggly path, run the EKF on noisy encoders and marker readings
    Returns:
    dict
    Logged errors "e" (N x 3),
    sigmas "sigma" (N x 3),
    "nees" and "nis"
    """
    rng = np.random.default_rng(seed)
    x_true = np.array([0.6,0.6,0]) #start
    ekf = EKF(x_true, np.diag([1e-4, 1e-4, 1e-4]), K_ENC, R_RB)
    log = {"e": [], "sigma":[], "nees": [], "nis":[]}

    for k in range(N_STEPS):
        #true wheel increments for a forward path with a weave
        v = 0.15
        w = 0.3 * np.sin(k / 200)
        wR = (v + w * WHEEL_BASE / 2) / WHEEL_RADIUS * T
        wL = (v - w *WHEEL_BASE / 2) / WHEEL_RADIUS * T

        #Move the robot with the model
        ds = WHEEL_RADIUS / 2 *(wR + wL)
        dth = WHEEL_RADIUS / WHEEL_BASE * (wR - wL)
        x_true += [ds * np.cos(x_true[2] + dth/2),
                   ds * np.sin(x_true[2] + dth/2), dth]
        x_true[2] = wrap(x_true[2])

        #predict from noisy encoders
        mR = wR + rng.normal(0,np.sqrt(K_ENC * abs(wR)))
        mL = wL + rng.normal(0,np.sqrt(K_ENC * abs(wL)))
        ekf.predict(mR,mL)

        #update from the marker every 5 steps
        #alternating 400 step windows of seen and hidden show the covariance
        # growing on odometry alone and collapsing when the marker returns
        seen = (k //400) % 2 == 0 and not hide_marker
        if k % 5 == 0 and seen:
            d = MARKER - x_true[:2]
            z = [np.hypot(*d) + rng.normal(0, np.sqrt(R_RB[0,0])),
                wrap(np.arctan2(d[1], d[0]) - x_true[2] + rng.normal(0, np.sqrt(R_RB[1,1])))]
            nu, S = ekf.update_range_bearing(z, MARKER)
            log["nis"].append(nu @ np.linalg.solve(S, nu))
        #log error against true, for evaluation only
        e = ekf.x - x_true
        e[2] = wrap(e[2])
        log["e"].append(e)
        log["sigma"].append(np.sqrt(np.diag(ekf.P)))
        log["nees"].append(e @ np.linalg.solve(ekf.P,e))
    return {key: np.array(val) for key, val in log.items()}
def compute_metrics(lg):
    """
    RMSE per state, mean NEES(should be ~3) and mean NIS (should be ~2)
    """
    rmse = np.sqrt((lg["e"]**2).mean(axis=0))
    return{"rmse_x": rmse[0],
            "rmse_y": rmse[1],
            "rmse_th": rmse[2],
            "nees": lg["nees"].mean(),
            "nis": lg["nis"].mean() if len(lg["nis"]) else np.nan}

if __name__ == "__main__":
    print(f"{'case':>14s}{'RMSE x':>10s}{'RMSE y':>10s}{'RMSE th':>10s}{'NEES':>8s}{'NIS':>8s}")
    for label, hide in [("marker seen", False), ("marker hidden", True)]:
        m = compute_metrics(simulate(hide_marker=hide))
        print(f"{label:>14s}{m['rmse_x']:10.4f}{m['rmse_y']:10.4f}{m['rmse_th']:10.4f}"
              f"{m['nees']:8.2f}{m['nis']:8.2f}")