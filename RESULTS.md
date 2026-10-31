# Results: what the system achieves, where it fails, and why

Sections 1 to 7 cover the classic ward, which uses the original 6 × 4 m room with a single delivery. These results are unchanged from the current code. Section 8 covers the paediatric ward, where the robot must visit several locations.

All results come from scripts in Analysis/ and can be reproduced using the instructions in the README. Each trial starts from one of 10 random positions near the pharmacy, with a random heading. The same starting positions are used for every test condition.

A trial is successful if the robot reaches the docking position within 5 cm and 5° of the target without colliding with anything. The simulator's true position is only used to measure performance and apply disturbances. The controller only uses the position estimated by the EKF.

## 1. Exec

Under normal conditions, the robot docks an average of 0.50 cm from the target, with a 95th-percentile error of 0.57 cm. All 10 trials are successful.

Without the camera (marker_never_seen), the robot relies on odometry alone. Its average docking error increases to 4.9 cm, and only 5 out of 10 trials succeed. This shows that the camera and EKF make docking around 10 times more accurate.

## 2. Robustness (Part E) 

The robot was tested under different conditions to see how well it handles obstacles, sensor noise, incorrect measurements and external disturbances.

Under normal conditions, the robot achieved a 100% success rate, with an average docking error of 0.50 cm and an EKF RMSE of 1.74 cm.

When a trolley was moved onto its planned route, the robot still succeeded in all 10 trials. However, it had to replan its path an average of 6 times, compared with 1.9 times under normal conditions. This shows that the LiDAR can detect obstacles and the planner can find alternative routes.

When the camera marker was hidden for the final metre, the robot relied on odometry for approximately the last 0.6 m. The average docking error increased slightly from 0.50 cm to 0.81 cm. This suggests that odometry can work well over short distances once the EKF has been corrected by the camera.

The robot also handled LiDAR dropouts, increased encoder noise and camera noise without any failed trials. Increasing encoder noise by five times raised the EKF RMSE from 1.74 cm to 2.44 cm before the marker was detected, but the camera helped correct the error afterwards.

An incorrect wheel radius caused more noticeable problems. When the wheel radius was wrong by 2%, the robot still succeeded in all 10 trials, but the average docking error increased to 1.35 cm. The NEES also increased to 7.34, showing that the EKF was too confident in its estimate. This suggests that an incorrect robot model can be more problematic than extra sensor noise, because the filter does not know its model is wrong.

Incorrect camera readings were another important test. One in every ten marker readings was deliberately made inaccurate. With the chi-squared filter enabled, all 10 trials succeeded, with an average docking error of 0.50 cm. The filter rejected around 14 readings per run. Without the filter, only 1 out of 10 trials succeeded, and the average docking error increased to 10.87 cm. Every failure was linked to an incorrect reading that the filter had not rejected.

The robot was also tested by applying a sideways push lasting 0.3 seconds, eight seconds into the route, before the camera marker was visible.

Small pushes of 6 N and 9 N did not affect the robot because they were below the tyres' grip limit of approximately 17 N. The robot maintained an average docking error of 0.50 cm.

A 12 N push caused some wheel slip, increasing the average docking error to 1.19 cm, with a maximum of 2.6 cm. The robot still succeeded in all 10 trials, although the EKF became less accurate because it could not detect the movement.

A stronger 15 N push caused every trial to fail. The robot rotated by approximately 20–30° without the encoders detecting the movement. This caused the position estimate to become incorrect, making it difficult for the robot to find the marker and return to its planned route.

Overall, the robot handled normal operating conditions and moderate disturbances well. Its main weakness was recovering from sudden movements that its sensors could not detect.


## 3. Failures

The script run_trials.py uses the recorded data to identify the cause of each failure. The file summary.md lists the failures, while the fail_<condition>_<n>.png images show what happened during each trial.

The first cause is losing track of the robot's position before the marker becomes visible. This happened in 4 out of 10 strong-push trials. When the wheels slip, the encoders may fail to detect the robot's rotation. Without the camera marker in view, there is no other sensor correcting the heading estimate. The EKF therefore believes the robot is following its planned route, even though it has rotated 20–30° off course. The camera then fails to see the marker because the robot is facing the wrong direction. In one trial, the robot hit the trolley while the estimated position was already 38 cm away from the true position.

The second cause is the filter rejecting correct camera readings. When a push occurs while the marker is visible, the EKF can become confident in an incorrect position estimate. The chi-squared filter may then reject correct camera readings because they differ too much from the predicted position.

To address this, a recovery method was added. If two readings are rejected consecutively, the EKF increases its estimated uncertainty, allowing it to accept readings that would otherwise be rejected. This improved two trials, reducing their errors from 19.9 cm to 4.3 cm and from 11.5 cm to 5.7 cm. However, this method cannot help if the marker is no longer visible.

The third cause is a limitation of using a single marker. Even after recovery, one trial finished 4.3 cm sideways from the target and 5.7° off in heading. These errors are related because rotating around a marker 0.4 m away by 5.7° produces approximately 4 cm of displacement. A single marker that measures range and bearing cannot fully detect this type of rotation. This is a limitation of the sensor setup rather than simply a tuning problem, as explained in Section 4 of MODEL.md.

The fourth cause occurs when the marker is briefly visible but then disappears. In two trials, the robot detected the marker for a short time before moving out of view. Once this happened, the EKF could no longer use the camera to correct the position estimate.

These failures show that the robot's main weakness is not its ability to follow a planned route, but its ability to recover when its position estimate becomes incorrect.

## 4. Sensing calibration (Part B) - `Analysis/marker_noise.png`

The camera was tested using 285 known robot positions, at distances between 0.4 m and 3.0 m and viewing angles between 0° and 60°.

The results showed that the range measurements had a consistent bias that increased with distance. At 1.5 m, the error was approximately +1.3 cm, while at 3 m the errors ranged from 0 to 11 cm and were much more spread out. This pattern suggested that the error was caused by a consistent pixel offset when calculating the marker's position using PnP.

To correct this, the error was fitted to the equation 0.0059 r², and this bias was removed from the measurements.

After calibration, range errors were approximately 2–6 mm at distances up to 1.8 m, while bearing error was around 0.1°. The original assumptions of 2 cm range error and 2° bearing error were therefore about 10 and 20 times too large, respectively.

The tests also showed that markers became harder to detect beyond approximately 2 m. At these distances, they were mainly detected when viewed almost straight on because the beds blocked angled views.

Blurred, noisy and darker images were detected at a similar rate to normal images, with 160 detections compared with 164 out of 285. However, the poorer images were less accurate at longer distances. Beyond 1.5 m, their RMS range error was 5.2 cm, compared with 3.8 cm before bias removal.

These results show why camera calibration is important. Correcting the measurements allows the EKF to estimate the robot's position more accurately, although image quality and viewing distance still affect performance.

## 5. Estimator consistency against accuracy (Part C) - `Analysis/consistency_sweep.png`

The EKF's process noise was adjusted to investigate how it affected estimation accuracy and confidence.

With no extra process noise, the NEES was 28.4, even though the EKF RMSE was only 1.11 cm and the docking error was 0.54 cm. Increasing the process noise to its original level reduced NEES to 5.1, with almost no change in accuracy.

Doubling the original process noise produced a NEES of 2.4, an EKF RMSE of 1.12 cm and a docking error of 0.50 cm. This setting was chosen because its NEES was closer to the ideal value of 3 while maintaining good accuracy.

Increasing the process noise to eight times its original value reduced NEES to 0.5, but the EKF RMSE and docking error remained almost unchanged.

The measurement noise matrix R was fitted, and three random seeds were tested for each setting.

Overall, the results show that changing the process noise can significantly affect how confident the EKF is in its estimate without noticeably improving the actual position accuracy.

The remaining error comes from information the robot cannot obtain from a single marker. Changing the filter settings cannot solve this problem. Using a marker that also provides orientation, adding a second marker, or introducing LiDAR-based localisation could provide the missing information.

## 6. Controller tuning (Part D) - `Analysis/weight_sweep.png`
The LQR controller was tested by scaling both Q matrices from 0.03 to 30 times their original Bryson's-rule values while keeping R constant. Three runs were completed for each setting.

At 0.03 times the original Q value, the estimated cross-track error was 0.36 cm, while the true cross-track error was 1.13 cm. The docking error was 1.09 cm.

Increasing Q to 0.3 times its original value reduced the docking error to 0.58 cm. At the original Bryson's-rule setting, the docking error decreased further to 0.50 cm.

Increasing Q to three times its original value only reduced the docking error slightly, to 0.47 cm. At 30 times the original value, the docking error reached 0.43 cm, but the RMS control correction increased from 0.037 to 0.106.

These results show that the controller is not the main source of error. Once Q reaches 0.3 times its original value, the estimated cross-track error remains around 0.3 cm, even as the controller works harder.

The difference between the estimated and true cross-track errors comes from the EKF. The controller can only respond to the estimated position, so it cannot completely correct errors that the estimator does not detect. As a result, the true tracking error stays close to 1 cm regardless of how much the controller gains are increased.

Bryson's-rule weights provide a good balance between accuracy and control effort. Lower weights result in worse docking accuracy, while much higher weights provide only a small improvement. On a real robot, excessive control effort could also increase wheel slip and battery consumption.

For these reasons, the original Bryson's-rule weights were kept.

The results also vary between random seeds. Since only three runs were completed for each setting, the results show general trends rather than precise differences in performance.

## 7. What these results do and do not support

First, the simulation makes the camera appear more reliable than it may be in real life. It does not include motion blur, lens distortion, rolling shutter or changing lighting conditions. Although camera noise was tested, real-world errors could still be larger.

Second, NIS is not a reliable way to validate camera noise in this simulation. MuJoCo produces the same image for the same robot position, meaning detection errors can behave like a consistent bias rather than independent random noise. Once the EKF has accounted for this bias, the differences between predicted and measured positions become very small. This explains why the NIS was only 0.07, compared with the ideal value of 2.

NEES is more useful in simulation because it compares the estimated position with the true position. With real hardware, NIS would provide more useful information about the consistency of the sensor measurements.

Another limitation is that a single marker cannot fully determine the robot's position and orientation. Rotation around the marker remains difficult to observe regardless of the route. Docking still works because the starting position is known and odometry drifts slowly, but sudden disturbances that rotate the robot are harder to correct.

The robot also lacks an absolute position correction before the marker becomes visible. During the first approximately 17 seconds, it mainly relies on odometry. Any disturbance during this period can go unnoticed. Although the LiDAR detects walls throughout the route, it is only used for mapping. Using LiDAR for localisation would be a logical next step.

The number of trials is another limitation. Ten trials are enough to show a clear difference between conditions with 100% and 10% success, but they are not enough to estimate reliability precisely. Even with 10 successful trials out of 10, the true success rate could be as low as approximately 74% at 95% confidence.

Finally, the map itself depends on the EKF's position estimate. If the estimate becomes incorrect, such as after a strong push, the map can become rotated or misaligned. This can then affect path planning as well.

Overall, the system performs well under normal conditions and handles several types of noise and obstacles. However, sudden disturbances remain a problem when the camera marker is unavailable or cannot provide enough information to correct the position estimate.

## 8. The paediatric ward - `Analysis/trials_paeds/`

The paediatric ward test uses a longer route with five stops. The robot starts near the pharmacy at a random position and heading. It then visits beds 1, 3, 4 and 2, parking in front of each bedside cabinet's ArUco marker (IDs 1–4). Finally, it returns to the charging dock (ID 7).

A round is only considered successful if the robot reaches all five stops within 5 cm and 5° of their targets without touching anything. Ten rounds were tested under normal conditions and with a moved trolley. Each round takes approximately 182 seconds of simulated time.

Under normal conditions, all 10 rounds succeeded, meaning all 50 stops were reached within 5 cm of their targets. The average docking error at the beds was 0.30 cm, with a worst error of 0.54 cm. At the charging dock, the average error was 0.75 cm, with a worst error of 1.01 cm. The EKF RMSE was 1.39 cm, and the robot replanned its route an average of 4.3 times per round.

When the trolley was moved, all 10 rounds still succeeded, with all 50 stops within 5 cm. Bed docking errors remained unchanged, while the average docking error at the charging dock was 0.74 cm. The EKF RMSE was 1.33 cm, and the average number of replans increased to 5.7.

Across all 100 stops, the mean docking error was 0.39 cm, the 95th-percentile error was 0.84 cm, and the worst error was 1.02 cm. The largest heading error was 1.5°.

One important finding is that multiple markers help the EKF maintain an accurate position estimate. The robot can use whichever markers are visible to correct its position, accepting around 1,000 readings per round. Even though the route is about three times longer than the classic delivery route, the EKF RMSE stays between 1.3 and 1.4 cm, compared with 1.7 cm in the classic ward. NEES also stays reasonably close to its ideal value of 3, ranging from 1.7 to 2.0.

The charging dock is less accurate than the beds. Its average docking error is around 0.74 cm, compared with 0.30 cm at the beds. The error is mainly sideways, with a heading error of approximately -1°. A 1° rotation around a marker 0.4 m away produces around 0.7 cm of displacement.

This matches the limitation discussed in Section 3. When approaching the charging dock along the aisle, the robot mainly sees the dock marker in front of it, making certain movements harder to detect. At the bedsides, the robot often has information from other markers viewed from different directions, which helps improve its position estimate.

The moved-trolley test also showed that the robot could adapt to changes in its surroundings. A trolley was moved across the route between beds 1 and 3 after the robot started. The map detected the obstruction, and the robot found another path. The number of replans increased from 4.3 to 5.7 per round, but all trials remained successful.

Camera calibration was also adjusted for the paediatric ward. The lighting was different, and the camera rendered images without anti-aliasing to improve simulation speed. The calibration script, Analysis/marker_noise.py, was therefore run again, producing the model in marker_noise_model_paeds.json. The model uses a bias of 0.0062 r² and a range-noise model of σ_r = 0.002 + 0.0044 r².

Testing all five markers at distances between 0.4 m and 2.5 m showed range errors below 1 mm at docking distance and around 5 mm at 1.6 m. Bearing error was 0.01°. The EKF only uses readings from markers it expects to be within 3 m, which is the calibrated operating range.

It is important to note that only the normal and moved-trolley conditions were tested in the paediatric ward. The more extensive robustness tests, including pushes, outliers and sensor noise, were carried out in the classic ward and are discussed in Sections 2 and 3.
