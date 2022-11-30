from gettext import npgettext
import sys
import sys
import threading
import time
import numpy as np
from threading import Thread

from zmq import VMCI_BUFFER_MAX_SIZE
from MBot.Messages.message_defs import mo_states_dtype, mo_cmds_dtype, mo_pid_params_dtype
from MBot.SerialProtocol.protocol import SerialProtocol
from rtplot import client
from scipy.signal import butter, lfilter
from simple_pid import PID
from pyPS4Controller.controller import Controller
import board
import adafruit_dotstar as dotstar

# CONTROL BANDWITH: 0.85943669 Degrees

JOYSTICK_SCALE = 32767

FREQ = 400
DT = 1/FREQ

RW = 0.0048
RK = 0.1210
ALPHA = np.deg2rad(45)

N_DOTS = 72
MAX_BRIGHTNESS = 0.055
MIN_BRIGHTNESS = 0.01

MAX_TILT = np.deg2rad(5) # Maximum inclination: 5 degrees
MAX_LINEAR_VELOCITY = 0.7# m/s --> Corresponds to duty cycle as for now.

MAX_DUTY = 0.8
MAX_TORQUE = 0.9

ARC_START = np.deg2rad(15)
ARC_STOP = 2*np.pi - np.deg2rad(15)

ARC = ARC_STOP - ARC_START
ARC_PER_DOT = ARC/N_DOTS
  
THETA_KP = 9  #8.0                  #9.0 @400hz
THETA_KI = 0.0
THETA_KD = 0.3   #0.4  #0.25   0.2          #0.08 @400hz

V_KP = 0#0.03  #0.01 @50hz  0.02@100hz
V_KI = 0.000000
V_KD = 0#0.001    #0.0005 @50hz 0.001@100

xbias = np.deg2rad(-1.5)#MAX_TILT*-0.3  #-0.2
ybias = 0#MAX_TILT*0.3   #0.2

AVG_WINDOW = 1
PSI_WINDOW = 10 #14
emf = 0.0636942675159*0.8


Theta_KF = 0#0.8
sleepT =0



velocity_command = 0,0

RAMP = 0.2 # time to reach max torque
RAMP_DELTA = MAX_TORQUE/(RAMP/DT) 
# ---------------------------------------------------------------------------
# Gray C. Thomas, Ph.D's Soft Real Time Loop
# This library will soon be hosted as a PIP module and added as a python dependency.
# https://github.com/UM-LoCoLab/NeuroLocoMiddleware/blob/main/SoftRealtimeLoop.py

"""
Soft Realtime Loop---a class designed to allow clean exits from infinite loops
with the potential for post-loop cleanup operations executing.

The Loop Killer object watches for the key shutdown signals on the UNIX operating system (which runs on the PI)
when it detects a shutdown signal, it sets a flag, which is used by the Soft Realtime Loop to stop iterating.
Typically, it detects the CTRL-C from your keyboard, which sends a SIGTERM signal.

the function_in_loop argument to the Soft Realtime Loop's blocking_loop method is the function to be run every loop.
A typical usage would set function_in_loop to be a method of an object, so that the object could store program state.
See the 'ifmain' for two examples.

Author: Gray C. Thomas, Ph.D
https://github.com/GrayThomas, https://graythomas.github.io
"""

import signal
import time
from math import sqrt

PRECISION_OF_SLEEP = 0.0001

# Version of the SoftRealtimeLoop library
__version__ = "1.0.0"

def clamp(input,lower,upper):
    if input <lower:
        return input
    if input > upper: 
        return upper
    return input

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
    # Mo :: Commands, States
    ser_dev.serializer_dict[101] = [lambda bytes: np.frombuffer(bytes, dtype=mo_cmds_dtype), lambda data: data.tobytes()]
    ser_dev.serializer_dict[121] = [lambda bytes: np.frombuffer(bytes, dtype=mo_states_dtype), lambda data: data.tobytes()]

def init_lights(brightness):
        dots = dotstar.DotStar(board.SCK, board.MOSI, N_DOTS, brightness=brightness)
        dots.fill(color=(0, 0, 0))
        dots.show()

        return dots

def compute_dots(roll, pitch):
        x = np.sin(roll)
        y = np.sin(pitch)

        slope = np.arctan(y/x)

        if y >= 0 and x >= 0:
                dot_position = np.pi/2 - slope
        elif y >= 0 and x <= 0:
                dot_position = 3/2 * np.pi - slope
        elif y <= 0 and x >= 0:
                dot_position = np.pi/2 - slope
        elif y <= 0 and x <= 0:
                dot_position = 3/2 * np.pi - slope

        dot_intensity = (abs(np.sin(roll)) + abs(np.sin(pitch)))/(2 * abs(np.sin(MAX_TILT)))
        center_dot = int((dot_position - ARC_START)/ARC_PER_DOT)
        half_dots = int(dot_intensity * N_DOTS/2)

        center_start = center_dot - half_dots
        center_stop = center_dot + half_dots + 1

        if center_start < 0:
                center_start = 0

        if center_stop > N_DOTS:
                center_stop = N_DOTS

        dots = np.arange(center_start, center_stop)
        return dots

# Our signal handler mycode
def signal_handler(signum, frame):  
    print("Signal Number:", signum, " Frame: ", frame)
    print("last sleep percentage ",sleepT/DT*100)
    print("Resetting Mo commands.")
    commands['kill'] = 1.0
    commands['motor_1_duty'] = 0.0
    commands['motor_2_duty'] = 0.0
    commands['motor_3_duty'] = 0.0
    ser_dev.send_topic_data(101, commands)
    exit(0)
     
 
def exit_handler(signum, frame):
    print('Exiting....')
    exit(0)




# ---------------------------------------------------------------------------

# Wheel rotation to Ball rotation transformation matrix

J11 = -2 * RW/(3 * RK * np.cos(ALPHA))
J12 = RW / (3 * RK * np.cos(ALPHA))
J13 = J12
J21 = 0
J22 = -np.sqrt(3) * RW/ (3 * RK * np.cos(ALPHA))
J23 = -1 * J22
J31 = RW / (3 * RK * np.sin(ALPHA))
J32 = J31
J33 = J31

#J = np.array([[J11, J12, J13], [J21, J22, J23], [J31, J32, J33]])
J = np.array([[J21, J22, J23], [J11, J12, J13], [J31, J32, J33]])

# Ball rotation Wheel contact point speed to transformation matrix

beta = np.pi/2,np.pi/2+np.pi*2/3,np.pi/2+np.pi*4/3   #wheel angles in order

IK = np.zeros((3,3))
for i in range (0,3):
        IK[i][0]=RK/RW*-np.cos(ALPHA)*np.cos(beta[i]) #Rx component to ith wheel
        IK[i][1]=RK/RW*-np.cos(ALPHA)*np.sin(beta[i]) #Ry
        IK[i][2]=RK/RW*np.sin(ALPHA)
#tentatively, seems like linear velocity of wheels is x to y when positive.
#FK = np.linalg.inv(IK)
FK = np.array([[RW/RK*0,RW/RK*-np.sqrt(6)/3,RW/RK*np.sqrt(6)/3],
                [RW/RK*-np.sqrt(2)*2/3,RW/RK*np.sqrt(2)/3,RW/RK*np.sqrt(2)/3],
                [RW/RK*np.sqrt(2)/3,RW/RK*np.sqrt(2)/3,RW/RK*np.sqrt(2)/3]])    
  
#rotation matrix around x. Seems like x "roll" comes after y "pitch"
def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
#rotation matrix around y
def Ry(theta):
  return np.matrix([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])
#rotation matrix around z
def Rz(theta):
  return np.matrix([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
# angular velocity tensor
def W(wx,wy,wz):
   return np.matrix([[0,-wz,wy],
                    [wz,0,-wx],
                    [-wy,wx,0]])
def forward_km(psi_1, psi_2, psi_3):
    psi = np.array([[psi_1], [psi_2], [psi_3]])

    # phi = J [3x3] * psi [3x1]
    phi = np.matmul(FK, psi)

    # returns phi_x, phi_y, phi_z
    return phi[0][0], phi[1][0], phi[2][0]

def compute_phi(psi_1, psi_2, psi_3):
    '''
    Parameters:
    ----------
    psi_1_counts: Encoder counts [MOTOR 1]
    psi_2_counts: Encoder counts [MOTOR 2]
    psi_3_counts: Encoder counts [MOTOR 3]

    Returns:
    --------
    phi_x: Ball rotation along x-axis (rad)
    phi_y: Ball rotation along y-axis (rad)
    phi_z: Ball rotation along z-axis (rad)
    '''

    # array
    psi = np.array([[psi_1], [psi_2], [psi_3]])

    # phi = J [3x3] * psi [3x1]
    phi = np.matmul(J, psi)

    # returns phi_x, phi_y, phi_z
    return phi[0][0], phi[1][0], phi[2][0]

# ---------------------------------------------------------------------------
#note that direction of Tx matches direction of roll.
def computeVel(droll,dpitch,dpsi1,dpsi2,dpsi3):
    dphi = compute_phi(dpsi1,dpsi2,dpsi3)
    
    return (droll+dphi[0])*RK,(dpitch+dphi[1])*RK
def computeVel2(droll,dpitch,dpsi1,dpsi2,dpsi3):
    dphi = np.matmul(FK,np.array([[dpsi1],[dpsi2],[dpsi3]]))
    return((dpitch+dphi[1][0])*RK,(droll+dphi[0][0])*RK)

def ramp(current,command):
    if np.abs(command)>np.abs(current):
        if current - command > RAMP_DELTA:
            return current -RAMP_DELTA
        if current - command < -RAMP_DELTA:
            return current +RAMP_DELTA  
    return command

if __name__ == "__main__":
    signal.signal(signal.SIGINT,signal_handler)
    imu_states = {'names': ['Roll', 'Pitch'],
                    'title': "Orientation",
                    'ylabel': "rad",
                    'xlabel': "time",
                    'colors' : ["r", "g"],
                    'line_width': [2]*2,
                    'yrange': [-2.0 * np.pi, 2.0 * np.pi]
                    }

    stability_controller = {'names': ['SP Body Roll', 'Body Roll', 'SP Body Pitch', 'Body Pitch'],
                    'title': "Stability Controller",
                    'ylabel': "rad",
                    'xlabel': "time",
                    'colors' : ["r", "g", "b", "y"],
                    'line_width': [2]*4,
                    'yrange': [-MAX_TILT, MAX_TILT]
                    }

    plot_config = [imu_states]
    client.initialize_plots(plot_config)

    ser_dev = SerialProtocol()
    register_topics(ser_dev)

    # Init serial
    serial_read_thread = Thread(target = SerialProtocol.read_loop, args=(ser_dev,), daemon=True)
    serial_read_thread.start()

    # Local structs
    commands = np.zeros(1, dtype=mo_cmds_dtype)[0]
    states = np.zeros(1, dtype=mo_states_dtype)[0]

    commands['kill'] = 0.0

    # Time for comms to sync
    time.sleep(1.0)

    # Send the gains 
    # ser_dev.send_topic_data(111, gains)
    ser_dev.send_topic_data(101, commands)

    theta_roll_sp = 0.0
    theta_pitch_sp = 0.0

    theta_roll_pid = PID(THETA_KP, THETA_KI, THETA_KD, theta_roll_sp)
    theta_pitch_pid = PID(THETA_KP, THETA_KI, THETA_KD, theta_pitch_sp)
    vx_pid = PID(V_KP,V_KI,V_KD,0)
    vy_pid = PID(V_KP,V_KI,V_KD,0)

    theta_roll_pid.output_limits = (-MAX_DUTY, MAX_DUTY)
    theta_pitch_pid.output_limits = (-MAX_DUTY, MAX_DUTY)
    #theta_pitch_pid.sample_time=DT
    #theta_roll_pid.sample_time=DT
    
    #dots = init_lights(MAX_BRIGHTNESS)

    #for t in SoftRealtimeLoop(dt=DT, report=True):
    startTime = time.time()
    roll = states['theta_roll']
    pitch = states['theta_pitch']

    psi_1 = states['psi_1']
    psi_2 = states['psi_2']
    psi_3 = states['psi_3']

    dpsi_1 =0
    dpsi_2 =0
    dpsi_3= 0

    dpsi_1list =[]
    dpsi_2list =[]
    dpsi_3list =[]

    vx,vy = 0,0
    vxlist = []
    vylist = []
    vxavg,vyavg =0,0

    T1p,T2p,T3p = 0,0,0

    n = 0
    k = 0
    while True:
        try:
            states = ser_dev.get_cur_topic_data(121)[0]
        except KeyError as e:
            print("<< CALIBRATING >>")
            #dots.fill(color=(255, 191, 0))
            #dots.show()
            continue
        #vx,vy = computePosDelta((states['theta_roll']-roll)/DT,(states['theta_pitch']-pitch)/DT,states['psi_1'],states['psi_2'],states['psi_3'])
        #vx,vy = computePosDelta(0,0,(states['psi_1']-psi_1)/DT,(states['psi_2']-psi_2)/DT,(states['psi_3']-psi_3)/DT)
        k+=1
        subfreq = 200
        k%=(FREQ/subfreq)  #100 hz
        if k == 0:
            #vx,vy = computeVel2((states['theta_roll']-roll)*subfreq,(states['theta_pitch']-pitch)*subfreq,(states['psi_1']-psi_1)*subfreq,(states['psi_2']-psi_2)*subfreq,(states['psi_3']-psi_3)*subfreq)
            vx,vy = computeVel2((states['theta_roll']-roll)*subfreq,(states['theta_pitch']-pitch)*subfreq,(states['dpsi_1']),(states['dpsi_2']),(states['dpsi_3']))

            vylist.append(vy)
            vxlist.append(vx)
            if len(vxlist)> AVG_WINDOW:
                vxlist.pop(0)
            if len(vylist)> AVG_WINDOW:
                vylist.pop(0)

           
           
            
            
            droll=(states['theta_roll']-roll)*subfreq
            
            #dpsi_1=(states['psi_1']-psi_1)*subfreq
            #dpsi_2=(states['psi_2']-psi_2)*subfreq
            #dpsi_3=(states['psi_3']-psi_3)*subfreq
            dpsi_1 = (states['dpsi_1'])
            dpsi_2 = (states['dpsi_2'])
            dpsi_3 = (states['dpsi_3'])


            dpsi_1list.append(dpsi_1)
            if len(dpsi_1list)> PSI_WINDOW:
                dpsi_1list.pop(0)
            
            dpsi_2list.append(dpsi_2)
            if len(dpsi_2list)> PSI_WINDOW:
                dpsi_2list.pop(0)
            
            dpsi_3list.append(dpsi_3)
            if len(dpsi_3list)> PSI_WINDOW:
                dpsi_3list.pop(0)


            roll = states['theta_roll']
            pitch = states['theta_pitch']

            psi_1 = states['psi_1']
            psi_2 = states['psi_2']
            psi_3 = states['psi_3']


            spx = clamp(vy_pid(np.average(vylist))+xbias,-MAX_TILT*0.6,MAX_TILT*0.6)
            spy = clamp(vx_pid(np.average(vxlist))+ybias,-MAX_TILT*0.6,MAX_TILT*0.6)
            theta_roll_pid.setpoint= spx
            theta_pitch_pid.setpoint = spy


       
        #if k==0:
           # vel_kp = 0.1
           # 

        vel_kp = 0.01  #7 0.14 0.01 seems to work?
        Tx = theta_roll_pid(states['theta_roll'])-Theta_KF*np.sin(states['theta_roll']-xbias)#-vel_kp*vy
        Ty = theta_pitch_pid(states['theta_pitch'])-Theta_KF*np.sin(states['theta_pitch']-ybias)#-vel_kp*vx
        #Tx = 0.6
        #Ty = 0

        Tz = 0.0

        # Motor 1-3's positive direction is flipped hence the negative sign

        T1= (-0.3333) * (Tz - (2.8284 * Ty))
        T2 = (-0.3333) * (Tz + (1.4142 * (Ty + 1.7320 * Tx))) 
        T3 = (-0.3333) * (Tz + (1.4142 * (Ty - 1.7320 * Tx)))

        Tmax = T1
        if(T2>T1): Tmax = T2
        if(T3>T2): Tmax = T3

        T1+=np.average(dpsi_1list)*emf
        T2+=np.average(dpsi_2list)*emf
        T3+=np.average(dpsi_3list)*emf

        if(Tmax> MAX_TORQUE):
            T1/=Tmax
            T2/=Tmax
            T3/=Tmax

        T1 = ramp(T1p,T1)
        T2 = ramp(T2p,T2)
        T3 = ramp(T3p,T3)

        T1p = T1
        T2p = T2
        T3p = T3
        
        commands['motor_1_duty'] = T1 
        commands['motor_2_duty'] = T2
        commands['motor_3_duty'] = T3

        ser_dev.send_topic_data(101, commands)

        data = [states['theta_roll'], states['theta_pitch']]
        client.send_array(data)

        if np.abs(states['theta_roll']) != 0.0:
            danger = compute_dots(states['theta_roll'], states['theta_pitch'])
        sleepT =DT-(time.time()-startTime)%DT
        time.sleep(sleepT)
        startTime = time.time()

        n+=1
        n %= (FREQ/5)
        if n==0:
            #print(forward_km(dpsi_1,dpsi_2,dpsi_3))
            #print(dpsi_1,dpsi_2,dpsi_3)
            print('vx ','{:+2.2f}'.format(vx) ,'vy ', '{:+2.2f}'.format(vy))#,'droll''{:+2.2f}'.format(droll),'dpsi_1','{:+2.2f}'.format(dpsi_1))
            #print('dpsi_1','{:+2.2f}'.format(dpsi_1))

        #print(states['theta_roll'])
        ##for dot in range(N_DOTS):
          ##  if dot in danger:
            ##        dots[dot] = (255, 20, 20)
            ## else:
               ##     dots[dot] = (53, 118, 174)
        #dots.show()

    print("Resetting Mo commands.")
    commands['kill'] = 1.0
    commands['motor_1_duty'] = 0.0
    commands['motor_2_duty'] = 0.0
    commands['motor_3_duty'] = 0.0
    ser_dev.send_topic_data(101, commands)

    #dots.fill(color=(0, 0, 0))
    #dots.show()
