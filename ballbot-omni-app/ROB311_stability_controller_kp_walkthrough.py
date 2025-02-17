import sys
import threading
import time
import numpy as np
from threading import Thread
from MBot.Messages.message_defs import mo_states_dtype, mo_cmds_dtype, mo_pid_params_dtype
from MBot.SerialProtocol.protocol import SerialProtocol
from DataLogger import dataLogger
import ps4_controller_api as ps4
import FIR as fir
import LPFS as lpfs
import differentiator as Diff


# ---------------------------------------------------------------------------
"""
ROB 311 - Ball-bot Stability Controller Walkthrough [Kp]

This program uses a soft realtime loop to enforce loop timing. Soft real time loop is a  class
designed to allow clean exits from infinite loops with the potential for post-loop cleanup operations executing.

The Loop Killer object watches for the key shutdown signals on the UNIX operating system (which runs on the PI)
when it detects a shutdown signal, it sets a flag, which is used by the Soft Realtime Loop to stop iterating.
Typically, it detects the CTRL-C from your keyboard, which sends a SIGTERM signal.

the function_in_loop argument to the Soft Realtime Loop's blocking_loop method is the function to be run every loop.
A typical usage would set function_in_loop to be a method of an object, so that the object could store program state.
See the 'ifmain' for two examples.

Authors: Senthur Raj, Gray Thomas, Yves Nazon and Elliott Rouse 
Neurobionics Lab / Locomotor Control Lab
"""

import signal
import time
from math import sqrt
from simple_pid import PID


PRECISION_OF_SLEEP = 0.0001

# Version of the SoftRealtimeLoop library
__version__ = "1.0.0"

class LoopKiller:
    def __init__(self, fade_time=0.0):
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGHUP, self.handle_signal)
        self._fade_time = fade_time
        self._soft_kill_time = None

    def handle_signal(self, signum, frame):
        self.kill_now = True

    def get_fade(self):
        # interpolates from 1 to zero with soft fade out
        if self._kill_soon:
            t = time.time() - self._soft_kill_time
            if t >= self._fade_time:
                return 0.0
            return 1.0 - (t / self._fade_time)
        return 1.0

    _kill_now = False
    _kill_soon = False

    @property
    def kill_now(self):
        if self._kill_now:
            return True
        if self._kill_soon:
            t = time.time() - self._soft_kill_time
            if t > self._fade_time:
                self._kill_now = True
        return self._kill_now

    @kill_now.setter
    def kill_now(self, val):
        if val:
            if self._kill_soon:  # if you kill twice, then it becomes immediate
                self._kill_now = True
            else:
                if self._fade_time > 0.0:
                    self._kill_soon = True
                    self._soft_kill_time = time.time()
                else:
                    self._kill_now = True
        else:
            self._kill_now = False
            self._kill_soon = False
            self._soft_kill_time = None



class SoftRealtimeLoop:
    def __init__(self, dt=0.001, report=False, fade=0.0):
        self.t0 = self.t1 = time.time()
        self.killer = LoopKiller(fade_time=fade)
        self.dt = dt
        self.ttarg = None
        self.sum_err = 0.0
        self.sum_var = 0.0
        self.sleep_t_agg = 0.0
        self.n = 0
        self.report = report

    def __del__(self):
        if self.report:
            print("In %d cycles at %.2f Hz:" % (self.n, 1.0 / self.dt))
            print("\tavg error: %.3f milliseconds" % (1e3 * self.sum_err / self.n))
            print(
                "\tstddev error: %.3f milliseconds"
                % (
                    1e3
                    * sqrt((self.sum_var - self.sum_err**2 / self.n) / (self.n - 1))
                )
            )
            print(
                "\tpercent of time sleeping: %.1f %%"
                % (self.sleep_t_agg / self.time() * 100.0)
            )

    @property
    def fade(self):
        return self.killer.get_fade()

    def run(self, function_in_loop, dt=None):
        if dt is None:
            dt = self.dt
        self.t0 = self.t1 = time.time() + dt
        while not self.killer.kill_now:
            ret = function_in_loop()
            if ret == 0:
                self.stop()
            while time.time() < self.t1 and not self.killer.kill_now:
                if signal.sigtimedwait(
                    [signal.SIGTERM, signal.SIGINT, signal.SIGHUP], 0
                ):
                    self.stop()
            self.t1 += dt
        print("Soft realtime loop has ended successfully.")

    def stop(self):
        self.killer.kill_now = True

    def time(self):
        return time.time() - self.t0

    def time_since(self):
        return time.time() - self.t1

    def __iter__(self):
        self.t0 = self.t1 = time.time() + self.dt
        return self

    def __next__(self):
        if self.killer.kill_now:
            raise StopIteration

        while (
            time.time() < self.t1 - 2 * PRECISION_OF_SLEEP and not self.killer.kill_now
        ):
            t_pre_sleep = time.time()
            time.sleep(
                max(PRECISION_OF_SLEEP, self.t1 - time.time() - PRECISION_OF_SLEEP)
            )
            self.sleep_t_agg += time.time() - t_pre_sleep

        while time.time() < self.t1 and not self.killer.kill_now:
            if signal.sigtimedwait([signal.SIGTERM, signal.SIGINT, signal.SIGHUP], 0):
                self.stop()
        if self.killer.kill_now:
            raise StopIteration
        self.t1 += self.dt
        if self.ttarg is None:
            # inits ttarg on first call
            self.ttarg = time.time() + self.dt
            # then skips the first loop
            return self.t1 - self.t0
        error = time.time() - self.ttarg  # seconds
        self.sum_err += error
        self.sum_var += error**2
        self.n += 1
        self.ttarg += self.dt
        return self.t1 - self.t0

# ---------------------------------------------------------------------------

def register_topics(ser_dev:SerialProtocol):
    ser_dev.serializer_dict[101] = [lambda bytes: np.frombuffer(bytes, dtype=mo_cmds_dtype), lambda data: data.tobytes()]
    ser_dev.serializer_dict[121] = [lambda bytes: np.frombuffer(bytes, dtype=mo_states_dtype), lambda data: data.tobytes()]

# ---------------------------------------------------------------------------

FREQ = 200
DT = 1/FREQ

RW = 0.0048
RK = 0.1210
ALPHA = np.deg2rad(45)

MAX_PLANAR_DUTY = 0.75 #0.8  
MAX_LEAN = np.deg2rad(3)   #prev 10  2.5
PHI_JOY_SCALE = 60 
PHI_TRIG_SCALE = 20  # max sum: 100. Conservative setting :60 


emf = 0.0636942675159*1.02   #1.1 to compensate for drag
bias = 0.0
gTorque = 0.33 # seems like maximum  0.4 when low battery
kA = 0.004

usePID = True
useFIR = False
compensateBackEmf = True#bad feature :( actually good now
compensateGravity = True
compensateAcceleration = False
velocityControl = True
feedForward = False  #also bad
logData =  True 
printData = True
# ---------------------------------------------------------------------------
# LOWPASS FILTER PARAMETERS

Fs = FREQ # Sampling rate in Hz
Fc = 100 # Cut-off frequency of the filter in Hz    100hz for lpf-s is mostly unjagged. see trial 12413   150  seems good. stopped oscillating. 120 graph shows reasonable
Fc_psi = 60  #tried 0.3.... not sure.  12 is good   30 works  20 works
Fc_phi = 20   #good is 7-8  20 also works with decreased kp, and works super good
Fn = Fc/Fs # Normalized equivalent of Fc
N = 60 # Taps of the filter


if(useFIR):
    lowpass_filter_x = fir.FIR()
    lowpass_filter_x.lowpass(N, Fn)
    lowpass_filter_y = fir.FIR()
    lowpass_filter_y.lowpass(N, Fn)

    #------- yeah...
    lowpass_psi1 =  lpfs.LPFS()
    lowpass_psi1.estimateGains(Fc,0.5)
    lowpass_psi2 =  lpfs.LPFS()
    lowpass_psi2.estimateGains(Fc,0.5)
    lowpass_psi3 =  lpfs.LPFS()
    lowpass_psi3.estimateGains(Fc,0.5)
else:
    lowpass_filter_x = lpfs.LPFS()
    lowpass_filter_x.estimateGains(Fc,0.5)
    lowpass_filter_y = lpfs.LPFS()
    lowpass_filter_y.estimateGains(Fc,0.5)

    lowpass_psi1 =  lpfs.LPFS()
    lowpass_psi1.estimateGains(Fc_psi,20)
    lowpass_psi2 =  lpfs.LPFS()
    lowpass_psi2.estimateGains(Fc_psi,20)
    lowpass_psi3 =  lpfs.LPFS()
    lowpass_psi3.estimateGains(Fc_psi,20)

lowpass_dphix = lpfs.LPFS()
lowpass_dphix.estimateGains(Fc_phi,2)

lowpass_dphiy = lpfs.LPFS()
lowpass_dphiy.estimateGains(Fc_phi,2)
thetaxDiff = Diff.Differentiator()
thetayDiff = Diff.Differentiator()
psi1Diff = Diff.Differentiator()
psi2Diff = Diff.Differentiator()
psi3Diff = Diff.Differentiator()

ffxRamp = Diff.SlewRateLimiter()
ffxRamp.maxDerivative = 0.1
ffyRamp = Diff.SlewRateLimiter()
ffyRamp.maxDerivative = 0.1

velRampJoy = 8
velRampTrig = 4
xRamp = Diff.SlewRateLimiter()
xRamp.maxDerivative = velRampJoy #1.5
yRamp = Diff.SlewRateLimiter()
yRamp.maxDerivative = velRampJoy #1.5 works conservatively  2.4 works  4 works


# ---------------------------------------------------------------------------
###############  THESE WILL NEED TO BE CAREFULLY ADJUSTED ##################
# --------------------------------------------------------------------------

# Proportional gains for the stability controllers (X-Z and Y-Z plane)

KP_THETA_X = 5.5    #7.5 -7.0   #10 has weird oscillation                               # Adjust until the system balances
KP_THETA_Y = 5.5                                  # Adjust until the system balances

# ---------------------------------------------------------------------------
#############################################################################
KP_v = 0.02  #0.05 @ 8hz  0.02 @20 hz   0.015@50hz   right now : 0.24   0.005@30hz
vx_pid = PID(KP_v,0,0.00,DT)   #0.002-0.003  0.001@20hz 0.0005@30hz
vy_pid = PID(KP_v,0,0.00,DT)
vx_pid.output_limits = (-MAX_LEAN,MAX_LEAN)
vy_pid.output_limits = (-MAX_LEAN,MAX_LEAN)

if(usePID):
    x_pid = PID(0, 0, 0, DT) #0.1  tall 
    y_pid = PID(0, 0, 0, DT) #0.1
    x_pid.Kd = 0.12#standard
    y_pid.Kd = 0.12  #0.
    vz_pid = PID(0.16,0,0.01,DT) # velocity gains
    z_pid = PID(0.4,0,0.08,DT)
    z_pid.output_limits = (-1,1)


# Wheel rotation to Ball rotation transformation matrix
J11 = 0
J12 = -np.sqrt(3) * RW/ (3 * RK * np.cos(ALPHA))
J13 = -1 * J12

J21 = -2 * RW/(3 * RK * np.cos(ALPHA))
J22 = RW / (3 * RK * np.cos(ALPHA))
J23 = J22

J31 = RW / (3 * RK * np.sin(ALPHA))
J32 = J31
J33 = J31

J = np.array([[J11, J12, J13], [J21, J22, J23], [J31, J32, J33]])

beta = np.pi/2,np.pi/2+np.pi*2/3,np.pi/2+np.pi*4/3
FK = np.zeros((3,3))
for i in range (0,3):
        FK[i][0]=RK/RW*-np.cos(ALPHA)*np.cos(beta[i]) #Rx component to ith wheel
        FK[i][1]=RK/RW*-np.cos(ALPHA)*np.sin(beta[i]) #Ry
        FK[i][2]=RK/RW*np.sin(ALPHA)
#tentatively, seems like linear velocity of wheels is x to y when positive.
IK = np.linalg.inv(FK)



# ---------------------------------------------------------------------------

def compute_motor_torques(Tx, Ty, Tz):
    '''
    Parameters:
    ----------
    Tx: Torque along x-axis
    Ty: Torque along y-axis
    Tz: Torque along z-axis

    Returns:
    --------
            Ty
            T1
            |
            |
            |
            . _ _ _ _ Tx
           / \
          /   \
         /     \
        /       \
       T2       T3

    T1: Motor Torque 1
    T2: Motor Torque 2
    T3: Motor Torque 3
    '''

    T1 = (-0.3333) * (Tz - (2.8284 * Ty))
    T2 = (-0.3333) * (Tz + (1.4142 * (Ty + 1.7320 * Tx)))
    T3 = (-0.3333) * (Tz + (1.4142 * (Ty - 1.7320 * Tx)))

    return T1, T2, T3

# ---------------------------------------------------------------------------

def compute_phi(psi_1, psi_2, psi_3):
    '''
    Parameters:
    ----------
    psi_1: Encoder rotation (rad) [MOTOR 1]
    psi_2: Encoder rotation (rad) [MOTOR 2]
    psi_3: Encoder rotation (rad) [MOTOR 3]

    Returns:
    --------
    phi_x: Ball rotation along x-axis (rad)
    phi_y: Ball rotation along y-axis (rad)
    phi_z: Ball rotation along z-axis (rad)
    '''

    # Converting counts to rad
    psi = np.array([[psi_1], [psi_2], [psi_3]])

    # phi = J [3x3] * psi [3x1]
    phi = np.matmul(J, psi)

    # returns phi_x, phi_y, phi_z
    return phi[0][0], phi[1][0], phi[2][0]

def transform_w2b(m1, m2, m3):
    """
    Returns Phi attributes
    """

    x = 0.323899 * m2 - 0.323899 * m3
    y = -0.374007 * m1 + 0.187003 * m2 + 0.187003 * m3
    z = 0.187003 * m1 + 0.187003 * m2 + 0.187003 * m3

    return x, y, z


if __name__ == "__main__":
    if(logData):
        trial_num = int(input('Trial Number? '))
        filename = 'ROB311_Stability_states%i' % trial_num
        filename2 = 'ROB311_Stability_effort%i' % trial_num
        filename3 = 'ROB311_Stability_filter%i' % trial_num
        dl = dataLogger(filename + '.txt')
        effortLogger = dataLogger(filename2 + '.txt')
        filterLogger = dataLogger(filename3 + '.txt')
    

    ser_dev = SerialProtocol()
    register_topics(ser_dev)

    #initialize ps4 controller
    rob311_bt_controller = ps4.ROB311BTController(interface="/dev/input/js0")
    rob311_bt_controller_thread = threading.Thread(target=rob311_bt_controller.listen, args=(10,))
    rob311_bt_controller_thread.daemon = True

    rob311_bt_controller_thread.start()


    # Init serial
    serial_read_thread = Thread(target = SerialProtocol.read_loop, args=(ser_dev,), daemon=True)
    serial_read_thread.start()

    # Local structs
    commands = np.zeros(1, dtype=mo_cmds_dtype)[0]
    states = np.zeros(1, dtype=mo_states_dtype)[0]


    psi_1 = 0.0
    psi_2 = 0.0
    psi_3 = 0.0

    # Motor torques
    T1 = 0.0
    T2 = 0.0
    T3 = 0.0

    # Desired theta
    desired_theta_x = np.deg2rad(0) #-1.6
    desired_theta_y = np.deg2rad(0)

    # Error in theta
    error_x = 0.0
    error_y = 0.0

    commands['kill'] = 0.0

    # Time for comms to sync
    time.sleep(1.0)

    ser_dev.send_topic_data(101, commands)

    print('Beginning program!')
    i = 0
    rob311_bt_controller.y_trim_count = 3   #36
    rob311_bt_controller.x_trim_count = 6    #4
    

    
    zeroed = False

    psi_1_start = 0.0
    psi_2_start = 0.0
    psi_3_start = 0.0


    for t in SoftRealtimeLoop(dt=DT, report=True):
        try:
            states = ser_dev.get_cur_topic_data(121)[0]
            if i == 0:
                t_start = time.time()
                if(not logData):
                    i = 1
            if(logData):
                i = i + 1
        except KeyError as e:
            continue
        if(logData):
            t_now = time.time() - t_start
       

        velocityControl = rob311_bt_controller.ltoggle
        # Define variables for saving / analysis here - below you can create variables from the available states in message_defs.py
        
        # Motor rotations
        psi_1 = states['psi_1']
        psi_2 = states['psi_2']
        psi_3 = states['psi_3']

        if(not zeroed):
            zeroed = True
            psi_1_start =psi_1
            psi_2_start =psi_2
            psi_3_start = psi_3

        #dpsi1 = states['dpsi_1']
        #dpsi2 = states['dpsi_2']
        #dpsi3 = states['dpsi_3']
        dpsi1 = psi1Diff.differentiate(psi_1)
        dpsi2 = psi2Diff.differentiate(psi_2)
        dpsi3 = psi3Diff.differentiate(psi_3)

        xRamp.maxDerivative = velRampJoy+velRampTrig*rob311_bt_controller.ltrigger
        yRamp.maxDerivative = velRampJoy+velRampTrig*rob311_bt_controller.ltrigger


        dpsi_1f = lowpass_psi1.filter(dpsi1)
        dpsi_2f = lowpass_psi2.filter(dpsi2)
        dpsi_3f = lowpass_psi3.filter(dpsi3)

        # #= psi1Diff.differentiate(psi_1f)
        #dpsi_2f #= psi1Diff.differentiate(psi_2f)
        #dpsi_3f #= psi1Diff.differentiate(psi_3f)


        # Body lean angles
        theta_x = (states['theta_roll'])+np.deg2rad(rob311_bt_controller.y_trim_count*0.05)
        theta_y = (states['theta_pitch'])-np.deg2rad(rob311_bt_controller.x_trim_count*0.05)

        phi_x, phi_y, phi_z = transform_w2b(psi_1,psi_2,psi_3)
        dphi_x,dphi_y,dphi_z = transform_w2b(dpsi1,dpsi2,dpsi3)
        
        dtheta_x = thetaxDiff.differentiate(theta_x)
        dtheta_y = thetayDiff.differentiate(theta_y)

        dphi_xf = lowpass_dphix.filter(dphi_x) # angular velocity of ball relative to ground
        dphi_yf = lowpass_dphiy.filter(dphi_y)

        

        xCommand = -np.deg2rad(4*rob311_bt_controller.ly)  #front/back
        yCommand = np.deg2rad(4*rob311_bt_controller.lx) #tz_demo_2
        
        if(velocityControl):
            phi_ycmd =  xCommand*(PHI_JOY_SCALE+rob311_bt_controller.ltrigger*PHI_TRIG_SCALE) #around 1 radian max, when 15   60 is stable.
            phi_xcmd =  yCommand*(PHI_JOY_SCALE+rob311_bt_controller.ltrigger*PHI_TRIG_SCALE)

            phi_xcmd = xRamp.limit(phi_xcmd)
            phi_ycmd = yRamp.limit(phi_ycmd)

            vy_pid.setpoint = phi_xcmd
            vx_pid.setpoint = phi_ycmd
            desired_theta_x = vx_pid(dphi_xf)
            desired_theta_y = vy_pid(dphi_yf)
        else:
            desired_theta_x = xCommand
            desired_theta_y = yCommand
            phi_xcmd = 0
            phi_ycmd = 0

        # Controller error terms
        error_x = desired_theta_x - theta_x
        error_y = desired_theta_y - theta_y
        # ---------------------------------------------------------
        # Compute motor torques (T1, T2, and T3) with Tx, Ty, and Tz

        # Proportional controller
        theta_xfd=lowpass_filter_x.filter(theta_x)
        theta_yfd=lowpass_filter_y.filter(theta_y)
        #theta_xfd = 0
        #theta_yfd = 0

        pEffortX = 0
        pEffortY = 0
        dEffortX = 0
        dEffortY = 0
        
        if(usePID):
            x_pid.setpoint = desired_theta_x
            y_pid.setpoint = desired_theta_y

            dEffortX = x_pid(theta_xfd)#*(1+np.abs(np.sin(theta_x)))
            dEffortY = y_pid(theta_yfd)#*(1+np.abs(np.sin(theta_y)))
            

            pEffortX = KP_THETA_X * error_x
            pEffortY = KP_THETA_Y * error_y
            Tx = dEffortX+ pEffortX
            Ty = dEffortY+ pEffortY
        else:
            Tx = KP_THETA_X * error_x
            Ty = KP_THETA_Y * error_y

        #Tz = -2.5*(rob311_bt_controller.lx)**3
        desired_z = (rob311_bt_controller.tz_demo_2)**3
        if(compensateBackEmf):
            #
            
            if(not rob311_bt_controller.rtoggle):
                z_pid.setpoint += desired_z*10*DT
                Tz = -z_pid((psi_1+psi_2+psi_3)-(psi_1_start+psi_2_start+psi_3_start))
            else:
                vz_pid.setpoint = desired_z*10 # velocity
                Tz= -vz_pid((dpsi_1f+dpsi_2f+dpsi_3f)/3)
                z_pid.setpoint = 0
                psi_1_start = psi_1
                psi_2_start = psi_2
                psi_3_start = psi_3
                z_pid.reset()
        else:
            Tz = -desired_z*2.5

        if(compensateGravity):
            Tx -= np.sin(theta_x)*gTorque
            Ty -= np.sin(theta_y)*gTorque

        phi_x_accel = np.tan(theta_x)*9.81/RK
        phi_y_accel = np.tan(theta_y)*9.81/RK
        if(compensateAcceleration):
            Tx -= kA*phi_x_accel
            Ty -= kA*phi_y_accel

        # ---------------------------------------------------------
        # Saturating the planar torques 
        # This keeps the system having the correct torque balance across the wheels in the face of saturation of any motor during the conversion from planar torques to M1-M3
        
        #maxXDuty = np.max([(1-np.abs(3*np.sin(theta_x))),0])
        #maxYDuty = np.max([(1-np.abs(3*np.sin(theta_y))),0])
        #maxDuty = np.hypot(maxXDuty,maxYDuty)*MAX_PLANAR_DUTY


        #if np.hypot(np.abs(Tx),np.abs(Ty)) > maxDuty:
           # Tx = Tx/maxDuty
           # Ty = Ty/maxDuty
        

       # if np.hypot(theta_x,theta_y) > MAX_LEAN:
       #     Tx = 0
       #     Ty = 0

        #T1,T2,T3 = 0,0,0
        T1, T2, T3 = compute_motor_torques(Tx, Ty, Tz)
        T1raw = T1
        T2raw = T2
        T3raw = T3

        
        if(feedForward):   
            phiCmd=np.array([[phi_xcmd],[phi_ycmd],[0]])
            psiTranslation=np.matmul(FK,phiCmd)
            
            ff1 = psiTranslation[0][0]*emf
            ff2 = psiTranslation[1][0]*emf
            ff3 = psiTranslation[2][0]*emf
        else:
            ff1 = 0
            ff2 = 0
            ff3 = 0

        #FEEDFORWARD
        if(feedForward):
            T1 += ff1
            T2 += ff2
            T3 += ff3

        # ---------------------------------------------------------
        impAvg = False
        if(compensateBackEmf):
            if(impAvg):
                if(T1!=0):
                    e1 = dpsi_1f*emf/T1
                else:
                    e1 = 0
                if(T2!=0):
                    e2 = dpsi_2f*emf/T2
                else:
                    e2 = 0
                if(T3!=0):
                    e3 = dpsi_3f*emf/T3
                else:
                    e3 = 0

                e = np.average([e1,e2,e3])

                T1*=np.max([(1+e),0])
                T2*=np.max([(1+e),0])
                T3*=np.max([(1+e),0])
            else:
                T1 += dpsi_1f*emf + bias
                T2 += dpsi_2f*emf + bias
                T3 += dpsi_3f*emf + bias
        #----------------------------------------------------

        
      
        largestTorque = np.max([np.abs(T1),np.abs(T2),np.abs(T3)])

        if(largestTorque>0.99):
            T1/=(largestTorque*1.1)
            T2/=(largestTorque*1.1)
            T3/=(largestTorque*1.1)
        

        
        # ---------------------------------------------------------
        #compute_phi(psi_1, psi_2, psi_3)

        # ---------------------------------------------------------
        if(printData):
            print("Iteration no. {}, T1: {:.2f}, T2: {:.2f}, T3: {:.2f}".format(i, T1, T2, T3))
        commands['motor_1_duty'] = T1
        commands['motor_2_duty'] = T2
        commands['motor_3_duty'] = T3  

        # Construct the data matrix for saving - you can add more variables by replicating the format below
        if(logData):
            data = [i] + [t_now] + [theta_x] + [theta_y] + [T1] + [T2] + [T3] + [phi_x] + [phi_y] + [phi_z] + [psi_1] + [psi_2] + [psi_3]
            dl.appendData(data)

            effort = [i]+ [t_now]+ [Tx] + [Ty] + [Tz] +[T1] + [T2] + [T3] +[T1raw]+[T2raw]+[T3raw] + [pEffortX] +[pEffortY] +[dEffortX] +[dEffortY] + [x_pid.setpoint] + [y_pid.setpoint] +[phi_xcmd] +[phi_ycmd] +[ff1] + [ff2]+[ff3]
            effortLogger.appendData(effort)

            filtering = [i] +[t_now] +[theta_xfd] + [theta_yfd] + [dpsi_1f] + [dpsi_2f] + [dpsi_3f] + [dphi_xf] + [dphi_yf]
            filterLogger.appendData(filtering)
        if(printData):
            print("Iteration no. {}, THETA X: {:.2f}, THETA Y: {:.2f}".format(i, theta_x, theta_y))
        ser_dev.send_topic_data(101, commands) # Send motor torques

    if(logData):
        dl.writeOut()
        effortLogger.writeOut()
        filterLogger.writeOut()

    print("Resetting Motor commands.")
    print("y_trim: ")
    print(rob311_bt_controller.y_trim_count)
    print("x_trim: ")
    print(rob311_bt_controller.x_trim_count)

    time.sleep(0.25)
    commands['motor_1_duty'] = 0.0
    commands['motor_2_duty'] = 0.0
    commands['motor_3_duty'] = 0.0
    time.sleep(0.25)
    commands['kill'] = 1.0
    time.sleep(0.25)
    ser_dev.send_topic_data(101, commands)
    time.sleep(0.25)

