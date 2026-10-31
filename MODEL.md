
This document explains how the robot is modelled, how its position is estimated, and how its movement is controlled. The calculations are checked using Python scripts in the Analysis folder. The robot frames are defined in frames.py so they stay consistent throughout the project.

1. Platform and Frames

The robot used in the project is the ROBOTIS TurtleBot3 Waffle Pi. It is a differential drive robot, meaning it moves using two driven wheels on the same axle and two passive caster wheels.

The main robot measurements are:

Wheel radius: 0.033 m

Distance between the driven wheels: 0.288 m

Maximum wheel speed: 7.88 rad/s, which gives a maximum linear speed of approximately 0.26 m/s

Robot footprint radius: 0.22 m

These values come from the robot model and are checked using the frames.check_robot_geometry function.

The project uses four main coordinate frames.

W is the world frame. Its origin is at the south west corner of the floor.

B is the robot base frame. Its origin is at the midpoint between the driven wheels. The x axis points forward and the y axis points to the left.

C is the camera frame.

L is the LiDAR frame.

M is the marker frame.

The camera and LiDAR are mounted at fixed positions on the robot. Their measurements are converted into the robot base frame using the known rotation and translation between the frames.

This makes sure that measurements from different sensors can be compared using the same coordinate system.

2. State Space Model of the Robot

The state space model describes the robot's position and orientation as it moves.

The robot state is represented by three values.

x is the robot's horizontal position in the world.

y is the robot's vertical position in the world.

theta is the direction the robot is facing.

Together, these form the robot pose.

The model uses the rotations measured by the wheel encoders as its inputs. These are the changes in rotation of the right and left wheels over one time step.

Using measured wheel rotation instead of the commanded motor speed means that errors in how well the motors follow their speed commands do not directly enter the model.

How the robot moves

When a wheel rotates, it moves the robot forward or backward depending on the direction of rotation. Assuming the wheels do not slip, the distance travelled by each wheel is its radius multiplied by its change in rotation.

The robot's forward movement is calculated by averaging the movement of the two wheels.

The change in heading is calculated from the difference between the right and left wheel movements divided by the distance between the wheels.

If both wheels rotate by the same amount, the robot moves straight. If one wheel rotates more than the other, the robot turns.

The model calculates the change in forward distance and heading using these relationships.

The new position is then calculated using the average heading during the time step. This is more accurate than using only the heading at the start because the robot follows a short curved path while turning.

The model updates every 20 milliseconds.

Because the movement equations contain sine and cosine terms, the model is nonlinear. The Extended Kalman Filter, or EKF, handles this by linearising the equations at each time step.

Process noise

The wheel encoders are not perfectly accurate, so the model includes process noise to represent uncertainty in the measured wheel rotations.

The uncertainty increases with the amount the wheels rotate. When the robot is stationary, this part of the model adds no uncertainty. When the robot moves further, the uncertainty increases.

The model also includes some extra process noise to account for real effects that are not fully represented by the equations.

The noise level was adjusted using Analysis/consistency_sweep.py. A scaling factor of 2 was selected because it made the uncertainty estimated by the EKF more closely match the actual errors seen in the simulation.

The resulting Normalised Estimation Error Squared, or NEES, value was approximately 2.4. The ideal value for this three state model is 3.

Assumptions and limitations

The model relies on several assumptions.

The wheels roll without slipping. This mostly holds in simulation, where odometry error is within 0.5 percent. However, hard acceleration or external pushes can cause the wheels to slip. In one test, a 15 N push rotated the robot by 25 degrees without producing a corresponding encoder signal.

The floor is flat and level. This is true in the simulation but would not necessarily hold on ramps or uneven surfaces.

The wheel radius and distance between the wheels are known accurately. This is true for the simulated robot because the values come from the model. In a real robot, tyre wear could change the effective wheel radius.

The robot's reference point is the midpoint between the driven wheels. The project checks this when the system starts.

Encoder noise is assumed to be random and to increase with wheel travel. Real encoders can also be affected by slipping and mechanical backlash.

The starting pose is assumed to be known because the robot is placed on the pharmacy pad. The initial uncertainty is set to 1 cm in position and 1 degree in heading.

This last assumption is especially important because the observability analysis shows that the sensors cannot determine every aspect of the robot's absolute pose by themselves.

3. Measurement Model

The camera detects an ArUco marker at a known world position of approximately x = 3.325 m and y = 3.399 m.

The camera estimates the marker's position relative to itself using a method called solvePnP. This measurement is then converted into the robot base frame.

The EKF uses two measurements from the marker.

Range is the distance between the robot and the marker.

Bearing is the angle between the robot's forward direction and the direction towards the marker.

The measurement model predicts what these two values should be based on the robot's estimated position and heading.

The EKF compares these predicted measurements with the actual camera measurements. It then uses the difference to update the estimated robot pose.

Calibrating marker noise

The measurement noise was calibrated using Analysis/marker_noise.py.

The robot was placed at 285 known positions and orientations. The tests covered distances between 0.4 m and 3.0 m, viewing angles between 0 and 60 degrees, and different marker positions within the camera image.

Each detected marker position was compared with the known position to measure the error.

The tests found that the range error was not simply random noise around zero. Instead, the measurements had a bias that increased with the square of the distance to the marker.

This happens because a small error in detecting the marker corners can create a larger position error when the camera calculates the marker's location. The fitted bias was 0.0059 multiplied by the range squared, measured in metres. At a distance of 1.5 m, this corresponds to approximately 1.3 cm.

The bias is removed from each range measurement before it is passed to the EKF.

After removing the bias, the remaining range error was approximately 2 to 6 mm at distances up to 1.8 m. The bearing error was approximately 0.1 degrees.

The calibrated noise model therefore uses a range uncertainty that increases with distance and a bearing uncertainty of 0.1 degrees.

The earlier assumptions of 2 cm range uncertainty and 2 degrees bearing uncertainty were much larger than the measured errors.

Beyond 2 m, the marker can only be seen from relatively direct viewing angles because the beds block views from the side.

4. Observability

Observability describes whether a system's internal state can be determined from its available measurements over time.

For this robot, the main question is whether the wheel encoders and the camera can determine the robot's absolute position and orientation.

Without the marker

During approximately the first 17 seconds of the route, the marker is not visible.

During this time, the wheel encoders provide information about how the robot moves, but they do not directly measure its absolute position in the world.

As a result, the robot's uncertainty grows as it moves.

One marker measurement

A single range and bearing measurement provides two pieces of information, while the robot pose contains three unknown values.

The measurement matrix has rank 2 rather than 3. This means one direction of the robot's pose cannot be determined from that measurement alone.

The important finding is that this limitation remains even when the robot receives repeated measurements while moving.

Observability along the route

The observability analysis is performed using Analysis/observability.py.

It checks how much information the measurements provide about the robot's state over a moving two second window.

The result shows that the system has rank 2 throughout the route rather than rank 3. One direction of the state remains unobservable.

That missing direction corresponds to rotating the robot's entire route around the marker.

Why does this happen?

Imagine rotating the entire robot route around the marker by a small angle.

The robot's position changes, and its heading changes by the same angle. However, the distance to the marker remains the same, and the bearing relative to the robot's own heading also remains the same.

The wheel encoder readings do not change either because they measure movement relative to the robot, not its absolute position in the world.

The sensors therefore cannot distinguish the original route from the rotated route.

The analysis demonstrated this by rotating a route by 6 degrees around the marker. The final position differed from the original route by 4.2 cm, yet the wheel, range and bearing readings agreed to within approximately 10 to the power of negative 15.

This confirms that a single marker providing range and bearing cannot fully determine the robot's absolute pose, regardless of how many measurements are collected along the route.

The known starting pose helps constrain the estimate because the robot begins from a known location and uses odometry to track its movement. However, errors can accumulate as the robot moves.

What this means for the robot

There are three important consequences.

First, the robot can still have a small docking error even when the marker is visible. The EKF's error settles at approximately 7 to 10 mm in the x direction and 0.6 to 0.9 degrees in heading. These errors are consistent with the unobservable rotation around the marker.

Second, the EKF can become overconfident. Because the filter uses a linear approximation of the measurement model, it can incorrectly behave as though it has learned something about the unobservable direction.

Without additional process noise, the NEES value rises to approximately 28 after the marker is detected, compared with an ideal value of 3.

Adding extra process noise makes the filter's uncertainty more realistic. However, it does not solve the underlying accuracy problem. The error remains around 1.1 cm because the sensors do not provide enough information to remove it.

Third, an external push can cause a significant failure. If a push rotates the robot without the wheel encoders recording the movement correctly, much of the resulting error can fall in the unobservable direction.

In one push trial, the robot finished with a heading error of 5.7 degrees and a sideways error of 4.3 cm. The analysis showed that this was consistent with the rotation around the marker.

The script run_trials.py can recognise this type of error automatically.

How could observability be improved?

Several possible improvements were identified, although they have not been implemented.

The first option is to measure the marker's orientation as well as its range and bearing. The solvePnP method already estimates orientation, so including it as another measurement could remove the rotation ambiguity.

The second option is to detect a second marker positioned away from the first. Measurements from two markers would provide additional geometric information about the robot's pose.

The third option is to use LiDAR scan matching against the map. The walls in the environment could provide additional reference points to help determine the robot's position and heading.

5. Control Model and Controllability

The robot uses an LQR tracker to follow its planned route. The controller is implemented in Control/lqr_tracker.py.

The controller compares the actual robot with a reference robot, sometimes called the ghost robot. The ghost represents where the robot should be and how it should be moving along the planned route.

The controller calculates the difference between the actual robot and the ghost using three error values.

The first error measures how far the ghost is ahead of the robot.

The second error measures how far the ghost is to the left of the robot.

The third error measures the difference between the ghost's heading and the robot's heading.

These errors are expressed relative to the robot's own coordinate frame. This makes the controller work with the robot's position and orientation rather than relying only on differences in world coordinates.

How the controller corrects errors

The LQR controller calculates changes to the robot's forward speed and turning speed relative to the reference robot.

The model is linearised around the current reference motion, assuming the errors are small.

The resulting equations describe how the three errors change over time in response to the controller's inputs.

One important feature is that the robot cannot move directly sideways. Its two driven wheels allow it to move forwards and turn, but there is no input that directly removes sideways error.

To correct sideways error, the robot must first change its heading and then move forwards. This is why heading control and forward movement work together to keep the robot on its planned route.

Controllability

Controllability describes whether the available control inputs can move the system between different states.

The project checks controllability using the controllability matrix, formed from the input matrix and its interactions with the system dynamics.

The results depend on how the reference robot is moving.

When the reference robot moves forwards at 0.15 m/s in a straight line, the system is fully controllable, with rank 3.

When the reference robot moves forwards at 0.15 m/s while turning at 0.5 radians per second, the system is also fully controllable, with rank 3.

When the reference robot turns on the spot at 0.5 radians per second, the model also has rank 3. Turning couples the position errors together, allowing the controller to influence both directions.

However, when the reference robot is completely stopped, with both forward and turning speeds equal to zero, the controllability rank falls to 2.

This means a stationary robot cannot directly correct a sideways position error using this linearised model.

Design choices based on controllability

The controllability results led to two important design decisions.

First, the planned route finishes with a straight 0.8 m approach to the bed. This allows the robot to correct its sideways position error while it is still moving forward, before it slows down and stops.

Second, the model is not linearised using a forward speed below 0.05 m/s. A minimum speed value is used so the Riccati equation can be solved using a model that remains controllable, even when the reference robot is stationary near the beginning or end of the route.

The model is converted into discrete time using a zero order hold method implemented with expm. This accounts for the controller updating every 20 milliseconds.

The controller gains are recalculated at each time step using the current reference speed and turning rate.

The error weights are selected using Bryson's rule, which sets each weight based on the maximum acceptable error.

Across the ward, the acceptable errors are 10 cm in each position direction and 17 degrees in heading.

Within 0.5 m of the bed, the limits become stricter, with 2 cm in each position direction and 6 degrees in heading.

The script Analysis/weight_sweep.py tests these choices by comparing tracking error and control effort.

Overall, the state space model describes how the robot moves, the EKF estimates its pose using wheel encoders and camera measurements, and the LQR controller corrects its movement errors. The observability analysis also identifies a fundamental limitation of using only one marker for range and bearing. The controllability analysis explains why the robot needs to correct sideways error while it is still moving, rather than waiting until it has stopped.