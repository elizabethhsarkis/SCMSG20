import numpy as np

WHEEL_RADIUS = 0.033 #wheel radius
WHEEL_BASE = 0.288 # wheel seperation

def wrap(angle):
    #keep angle from -pi to pi
    return (angle + np.pi) % (2 * np.pi) - np.pi

class EKF:
    """
    Extended Kalman filter for the TurtleBot3 pose.
    Prediction uses wheel encoder increments, update uses range and bearing
    to a marker at a known location
    """
    def __init__(self, x0, P0, k_enc, R_rb, r = WHEEL_RADIUS, b = WHEEL_BASE):
        self.x = np.array(x0, dtype=float) # x,y,theta in world frame
        self.P = np.array(P0, dtype=float)
        self.k_enc = k_enc #encoder variance per rad
        self.R_rb = np.array(R_rb) #2x2 range/bearing noise
        self.r = r
        self.b = b

    def predict(self, dphi_R, dphi_L):
        """
        Propagate the state and covariance through one step of wheel motion.
        Parameters:
        dphi_R and dphi_L as a float
        wheel angle change over the step (in radians).
        """
        r = self.r
        b = self.b
        #distance and heading change from increments
        ds = r / 2 * (dphi_R + dphi_L)
        dth = r / b * (dphi_R - dphi_L)
        #midpoint heading, turns during the step, moving along theta + dth/2
        #is closer than just moving along theta
        phi = self.x[2] + dth /2
        cos = np.cos(phi)
        sin = np.sin(phi)
        self.x = self.x + np.array([ds * cos, ds * sin, dth])
        self.x[2] = wrap(self.x[2])
        #jacobians, with respect with the state and the wheel increments
        #with F being the state and L being the increment
        F = np.array([[1,0,-ds * sin],
                      [0,1, ds * cos],
                      [0,0,1]])
        L = np.array([[r/2*cos - ds*sin*r/(2*b),
                       r/2*cos + ds*sin*r/(2*b)],
                       [r/2*sin + ds*cos*r/(2*b),
                        r/2*sin - ds*cos*r/(2*b)],
                        [r/b,-r/b]])
        #process noise from the encoders. Variance grows with distance
        #turned, parked robot gains no uncertainty, the small floor keeps Su positive
        Su = np.diag([self.k_enc * abs(dphi_R) + 1e-9,
                      self.k_enc * abs(dphi_L) + 1e-9])
        self.P = F @ self.P @ F.T + L @ Su @ L.T

        
    def update_range_bearing(self,z,marker_xy):
        """
        Correct the state with the marker detection
        Parameters:
        z = array measured from the base frame (range, bearing)
        marker_xy = array marker position in the world frame

        returns:
        (ndarray, ndarray)
        Innovation and covariance, for NIS test.
        """
        #prediced measurement from estimate
        dx = marker_xy[0] - self.x[0]
        dy = marker_xy[1] - self.x[1]
        q = dx**2 + dy**2
        rng = np.sqrt(q)
        z_hat = np.array([rng, wrap(np.arctan2(dy,dx) - self.x[2])])
        #measurement jacobian
        H = np.array([[-dx / rng, - dy / rng, 0],
                      [dy / q, -dx / q, -1]])
        #Innovation and gain, 
        #bearing innovation is wrapped to not throw off heading
        nu = np.array(z) - z_hat
        nu [1] = wrap(nu[1])
        S = H @ self.P @ H.T + self.R_rb
        K = self.P @ H.T @ np.linalg.inv(S)

        #state + covariance update
        #joseph form keeps P symmetric and positive even with rounding
        self.x = self.x + K @ nu
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_rb @ K.T
        return nu, S