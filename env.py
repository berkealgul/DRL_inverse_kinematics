import numpy as np
from math import cos, sin

"""
    NOTE: This code assumes the arm has 6 joints and fixed part lenght 
    Change in arm model will lead wrong results. Dont forget to update
    if you change the arm 
"""

class ArmEnvironment:
    def __init__(self):
        self.L1 = 7.0
        self.L2 = 26.5
        self.L3 = 27.3
        self.L4 = 7.0
        self.L5 = 20.0
        
        self.observation_space = (12,)
        self.action_space = [6,]
        self.angular_speed = 1.5 * np.pi / 180
        self.action_range = (-self.angular_speed, self.angular_speed)


        self.bounds = np.array([0.7,np.pi/2,0.9,np.pi/2,np.pi/2,np.pi/2], dtype=np.float32)
        self.initial_angles = None
        self.angles = None
        self.target_ee = None

        """
            status = 0 -> episode continues
            status = 1 -> episode ended. Task was successful
            status = -1 -> episode ended. Task was failure
        """
        self.status = 0 
        self.distace_tresh = 1

        # reward constants
        self.k = 20
        self.a = 1 / 70
        self.b = 5 / np.pi
        self.t = 100

        self.reset()

    def reset(self):
        self.status = 0
        arm_radius = 60 # maximum range of ee   
        self.target_ee = arm_radius*(2*np.random.rand(3)-1)
        self.target_ee[2] = np.absolute(self.target_ee[2]) # z must be non negative
        self.target_ee[0] = np.absolute(self.target_ee[0]) # x must be non negative
        self.initial_angles =  self.bounds*(2*np.random.rand(6)-1)
        self.angles = np.copy(self.initial_angles)
        ee = self.Dof_6()
        state = np.concatenate((self.angles, ee, self.target_ee), axis=0)
        return state

    def step(self, action):
        if self.status != 0:
            print("Episode is over! Requires restart")
            return None, None, True, "End Status: "+str(self.status)

        self.angles = self.angles + action
        ee = self.Dof_6()
        dist = self.update_status(ee)
        reward = self.calculate_reward(dist)
        new_state = np.concatenate((self.angles, ee, self.target_ee), axis=0)
        
        done = False
        if self.status != 0:
            done = True

        info = "ee: "+str(ee)

        return new_state, reward, done, info

    def update_status(self, ee):
        dist = np.subtract(self.target_ee, ee)
        dist = np.sqrt(dist.dot(dist))
        if dist <= self.distace_tresh:
            self.status = 1

        angles = np.absolute(self.angles)
        isFailed = np.any(np.greater(angles, self.bounds))
        if isFailed:
            self.status = -1

        return dist


    def calculate_reward(self, ee):
        dist =  np.subtract(self.target_ee, ee)
        dist = np.sqrt(dist.dot(dist))

        if self.status == -1:
            return -self.k
        
        dAngles = np.subtract(self.initial_angles, self.angles)
        dA = np.sqrt(dAngles.dot(dAngles))

        r = -self.a*dist - self.b*dA
        if self.status == 1:
            r += self.t

        return r

    def Dof_6(self):
        t0 = self.angles[0]
        t1 = self.angles[1]
        t2 = self.angles[2]
        t3 = self.angles[3]
        t4 = self.angles[4]

        px = self.L4*(cos(t0)*cos(t1)*sin(t2) + cos(t0)*cos(t2)*sin(t1)) - self.L5*(cos(t4)*(sin(t0)*sin(t3) + cos(t3)*(cos(t0)*sin(t1)*sin(t2) - cos(t0)*cos(t1)*cos(t2))) - sin(t4)*(cos(t0)*cos(t1)*sin(t2) + cos(t0)*cos(t2)*sin(t1))) - self.L2*cos(t0)*sin(t1) + self.L3*cos(t0)*cos(t1)*cos(t2) - self.L3*cos(t0)*sin(t1)*sin(t2)
        py = self.L5*(cos(t4)*(cos(t0)*sin(t3) - cos(t3)*(sin(t0)*sin(t1)*sin(t2) - cos(t1)*cos(t2)*sin(t0))) + sin(t4)*(cos(t1)*sin(t0)*sin(t2) + cos(t2)*sin(t0)*sin(t1))) + self.L4*(cos(t1)*sin(t0)*sin(t2) + cos(t2)*sin(t0)*sin(t1)) - self.L2*sin(t0)*sin(t1) + self.L3*cos(t1)*cos(t2)*sin(t0) - self.L3*sin(t0)*sin(t1)*sin(t2)
        pz = self.L1 - self.L4*(cos(t1)*cos(t2) - sin(t1)*sin(t2)) + self.L2*cos(t1) - self.L5*(sin(t4)*(cos(t1)*cos(t2) - sin(t1)*sin(t2)) - cos(t3)*cos(t4)*(cos(t1)*sin(t2) + cos(t2)*sin(t1))) + self.L3*cos(t1)*sin(t2) + self.L3*cos(t2)*sin(t1)

        return np.array([px,py,pz], dtype=np.float32)
