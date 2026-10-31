INSTALL

Install the required Python packages by running:
pip install -r requirements.txt


THE TWO WARDS

The project has two different ward layouts.

Paediatric ward

This is the default ward. You can also select it using --ward paeds.

The paediatric ward is an 8 × 6 m room with four beds. Each bed has a cot frame, duvet, pillow, teddy bear and name plate. There is also a bedside cabinet with an ArUco marker at each bed, a pharmacy counter, a charging dock with its own marker, toys, plants, a wall mural and windows.

The robot completes a full medication round by travelling from the pharmacy to bed 1, bed 3, bed 4 and bed 2, before returning to the charging dock. It stops in front of each marker along the way.

Classic ward

Select this ward using --ward classic.

This is the original 6 × 4 m room with a single bedside marker and one delivery destination. All results in RESULTS.md come from this ward and can be reproduced using the current code.

The --ward option works with every script. You can also select a ward by setting the WARD environment variable.

The paediatric ward layout is defined in ward_paeds.py, while the classic ward is defined in Build_scene.py.

RUNNING THE DEMO

Run these commands from the project folder.

Generate both ward layouts and their required files by running:
python Build_scene.py --all


This only needs to be done once. It generates the scenes, robot files, marker images and decorations for both wards.

To run the full medication round and watch the robot in the MuJoCo viewer, use:
python run_delivery.py --view


To run the same demonstration without opening the viewer, use:
python run_delivery.py


This prints the robot's progress at each stop and saves the results as delivery_result.png.

To test how the robot handles a moved trolley, run:
python run_delivery.py --condition moved_trolley --view


To run the original single delivery in the classic ward, use:
python run_delivery.py --ward classic --view


CREATING VIDEOS AND IMAGES

The following commands generate images and videos showing the robot in action.

To create overview images of the ward, run:
python make_video.py --stills


This generates media/ward_overview.png, media/ward_corner.png and media/ward_top.png.

To create a video of the full medication round, run:
python make_video.py


This saves the video as media/paeds_round.mp4 and plays the round at three times the normal speed.

For a quicker preview with smaller images, no shadows and four times the normal speed, run:
python make_video.py --quick


To create a video showing the robot handling a moved trolley, run:
python make_video.py --condition moved_trolley


The video follows the robot as it moves through the ward. It shows the planned route in green and the robot's estimated position as a blue ring. It also shows when the camera detects a marker, what the robot's camera sees, the live map and the robot's progress through the medication round.

The video requires imageio and imageio_ffmpeg to create an H.264 MP4 file. If these packages are unavailable, it uses OpenCV's mp4v codec instead.

On a computer with a graphics card, generating the video should take a few minutes.

REPRODUCING THE ANALYSIS

Each analysis script saves its results, figures and data in the Analysis/ folder.

The studies described in RESULTS.md use the classic ward. Run these scripts with --ward classic to reproduce the results. They will not run in the paediatric ward, which prevents the original results from being overwritten.

Camera calibration

Run:
python Analysis/marker_noise.py --ward classic


This tests how marker measurement errors change with distance and viewing angle. It also generates marker_noise_model.json.

The script takes around one minute using two CPU cores.

Observability analysis

Run:
python Analysis/observability.py --ward classic


This checks which parts of the robot's position can be estimated along the route and identifies any directions that cannot be observed. It takes a few seconds.

EKF consistency testing

Run:
python Analysis/consistency_sweep.py --ward classic


This tests how additional process noise affects the consistency of the Extended Kalman Filter using NEES and NIS. It takes around 10 minutes using two CPU cores.

LQR controller testing

Run:
python Analysis/weight_sweep.py --ward classic


This tests the relationship between tracking accuracy and control effort. It takes around 12 minutes using two CPU cores.

Robustness testing

Run:
python Analysis/run_trials.py --ward classic


This tests every robustness condition using 10 random starting positions and records the causes of any failures. It takes around 50 minutes using two CPU cores.

Paediatric ward analysis

The paediatric ward has its own camera calibration and delivery tests.

To calibrate the camera for the paediatric ward, run:
python Analysis/marker_noise.py


This generates marker_noise_model_paeds.json, which accounts for the different lighting conditions in the room.

To test the full medication round under normal conditions and with a moved trolley, run:
python Analysis/run_trials.py --conditions nominal moved_trolley


The results are saved in Analysis/trials_paeds/. Each round takes approximately 1.5 minutes using two CPU cores.

For more information, MODEL.md explains the mathematical model, its assumptions and the observability and controllability analysis. RESULTS.md contains the test results and explains what they show and where the system has limitations.

HOW THE SYSTEM WORKS

The robot updates its sensors, position estimate and movement plan at different time intervals.

Every 20 ms, run_delivery.py reads the wheel encoders and uses their measurements to predict the robot's position through the EKF.

Every 100 ms, the robot checks for ArUco markers when they are expected to be visible. Whenever a marker is detected, the EKF uses the measurement to correct its position estimate.

Every 200 ms, the robot adds a new LiDAR scan to its map. If an obstacle blocks the planned route, the planner finds another path.

The LQR controller then uses the EKF's position estimate to decide how fast the robot should move and turn. In the paediatric ward, this process repeats at each stop until the medication round is complete.

The simulator's true position is only used to record results, measure performance and create videos. The controller relies on the EKF's estimated position rather than the true position.

PROJECT STRUCTURE AND RESPONSIBILITIES

The project is divided into three main parts, with each person responsible for a different part of the system.


The Sensing/ folder handles the sensor inputs, including the LiDAR occupancy map and ArUco marker detection. It supports detecting one or several markers.


The Estimation/ folder contains the Extended Kalman Filter. It uses wheel encoder readings to predict the robot's position and camera measurements of marker range and bearing to correct the estimate.


The Control/ folder handles path planning using A*, the reference trajectory for the robot, LQR tracking and the sequence of stops in the medication round.

Ward setup

The files Build_scene.py, ward_paeds.py and ward_select.py define the two ward layouts and control which one is selected.

Video generation

The file make_video.py creates the videos and images used to show the robot's movement and performance.

In the paediatric ward, all decorations are placed in MuJoCo geometry group 2, with collisions disabled. The LiDAR also ignores this group. This means decorations such as toys, plants and the mural do not affect the robot's sensor readings or movement.