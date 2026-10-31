import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HIDE = (4,7) #same window as run function


def plot_errors(log_file = ROOT / "ekf_log.npy"):
    """
    Plot the EKF error in x, y and theta againt time with +- 3 sigma bound
    log columns: t, e_x, e_y, e_th, sigma_x, sigma_y, sigma_th
    """
    data = np.load(log_file)
    t = data[:,0]
    e = data[:,1:4]
    sigma = data[:,4:7]

    #theta in degrees
    scale = [100, 100, np.degrees(1)]
    labels = ["x error (cm)", "y error (cm)", "heading error (deg)"]
    fig, axes = plt.subplots(3,1, sharex = True, figsize = (8,7))
    for i, ax in enumerate(axes):
        #filter keeps error inside the 3 sigma band 
        ax.fill_between(t, -3* sigma[:,i] * scale[i], 3 * sigma[:,i] * scale[i],
                        color = "blue", alpha = 0.2, label="+-3 sigma")
        ax.plot(t, e[:,i] * scale[i], color = "blue", label = "error")
        # shade where detections were ignored
        ax.axvspan(HIDE[0], HIDE[1], color = "grey", alpha = 0.2, label="marker hidden")
        ax.axhline(0, color= "black", linewidth = 0.5)
        ax.set_ylabel(labels[i])
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("EKF estimation error in the ward scene")
    fig.tight_layout()
    fig.savefig(ROOT / "ekf_errors.png", dpi = 150)
    plt.show()
    #fraction of samples inside the bounds
    inside = np.mean(np.abs(e) <= 3 * sigma, axis = 0) * 100
    print(f"Inside 3 sigma (%): x{inside[0]:.1f}, y {inside[1]:.1f}, theta {inside[2]:.1f}")

if __name__ == "__main__":
    plot_errors()