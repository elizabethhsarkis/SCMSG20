# robustness conditions
#each entry changes some fields of run_delivery.TrialConditions. every condition
#also gets random start poses near the pharmacy (see run_trials.py), so "nominal"
#is already varied

from Build_scene import TROLLEY_MOVE_TO

CONDITIONS = {
    #the baseline
    "nominal": {},
    #a nurse wheels the trolley across the route just as the robot sets off. classic ward: (2.3, 1.5)
    #sits on the planned route and forces a 0.3-0.6 m detour. paediatric ward: (3.3, 3.0), across the
    #line between bed 1 and bed 3. the lidar map also has to clear the old spot
    "moved_trolley": {"move_trolley_to": TROLLEY_MOVE_TO},
    #someone stands in front of the marker for the last metre: dock on odometry alone
    "marker_hidden": {"marker_hidden_within_m": 1.0},
    #the marker is never seen at all: the whole trip on odometry (the baseline the camera improves on)
    "marker_never_seen": {"marker_hidden_within_m": 100.0},
    #1 in 5 lidar beams return nothing (dark or shiny surfaces)
    "lidar_dropout": {"lidar_dropout": 0.2},
    #encoders 5x noisier than the EKF thinks (worn wheels, carpet)
    "encoder_noise_x5": {"enc_noise_scale": 5.0},
    #wheels 2% smaller than the model says (tyre wear) - a model error, not noise
    "wheel_radius_2pct": {"wheel_radius_error": 0.02},
    #sideways bumps of increasing strength mid-route (6 N is below the tyres' grip, ~17 N)
    "push_gentle": {"push_force_n": 6.0, "push_at_s": 8.0, "push_duration_s": 0.3},
    "push_9N": {"push_force_n": 9.0, "push_at_s": 8.0, "push_duration_s": 0.3},
    "push_12N": {"push_force_n": 12.0, "push_at_s": 8.0, "push_duration_s": 0.3},
    #a hard sideways shove mid-route (bumped by a passer-by). the wheels slip, so odometry misses it
    "push": {"push_force_n": 15.0, "push_at_s": 8.0, "push_duration_s": 0.3},
    #same push without gate recovery - shows the gate locking out good readings
    "push_no_recovery": {"push_force_n": 15.0, "push_at_s": 8.0, "push_duration_s": 0.3,
                         "gate_recovery": False},
    #grainy, blurred, darker camera image
    "camera_noise": {"camera_noise": True},
    #1 in 10 marker readings badly wrong, gate ON
    "outliers_gated": {"outlier_rate": 0.1, "gating": True},
    #same outliers, gate OFF - shows what the chi-squared gate is worth
    "outliers_ungated": {"outlier_rate": 0.1, "gating": False},
}
