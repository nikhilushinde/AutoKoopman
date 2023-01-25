#!/usr/bin/env python3.7
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
import pickle
import sys
import yaml
import time
from casadi import *
from scipy.io import loadmat

import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Bool



""" --------------------------------------------------------------------------------------------------------------------
Auxiliary Functions
-------------------------------------------------------------------------------------------------------------------- """

def process_lidar_data(data):
    """convert the lidar measurement to a 2-dimensional point cloud"""

    # parameter (defined by the lidar that is used)
    phi = -2.35619449615                                # initial angle
    delta_phi = 0.00436332309619                        # angle increment

    # loop over all lidar beams
    points = np.zeros((2, len(data)))

    for i in range(0, len(data)):
        points[0, i] = data[i] * np.cos(phi)
        points[1, i] = data[i] * np.sin(phi)
        phi = phi + delta_phi

    return points[:, 270:-270] + np.array([[0.1], [0.0]])

def transformation_local_frame(x, y, phi, x_goal):
    """transform the goal state from the global to the local coordinate frame"""

    T = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

    return np.dot(T, np.resize(x_goal, (2, 1)) - np.array([[x], [y]]))

def init_optimizer(A_, B_, steps, horizon, nlidar, width, length):
    """initialize the casadi optimizer"""

    # initialization
    nx = A_.shape[0]
    nu = B_.shape[1]

    opti = casadi.Opti()

    # compute overall propagation matrices
    A = A_ 
    B = B_

    for i in range(horizon - 1):
        A = A @ A_
        B = A_ @ B + B_

    # initialize variables
    x = opti.variable(nx, steps + 1)
    u = opti.variable(nu, steps)
    x0 = opti.parameter(nx, 1)
    xf = opti.parameter(2, 1)
    points = opti.parameter(2, nlidar)

    # define cost function
    cost = 5*(x[0, steps] - xf[0])**2 + (x[1, steps] - xf[1])**2

    opti.minimize(cost)

    # system dynamics constraints
    for i in range(steps):
        opti.subject_to(x[:, i+1] == mtimes(A, x[:, i]) + mtimes(B, u[:, i]))

    opti.subject_to(x[:, 0] == x0)

    # collision avoidance constraints
    for i in range(1, steps+1):
        for j in range(points.shape[1]):

            opti.subject_to((points[0, j] - x[0, i])**2 + (points[1, j] - x[1, i])**2 >=
                            (length/2)**2 + (width/2)**2)

    # input constraints
    opti.subject_to(u[0, :] <= 0.4)
    opti.subject_to(u[0, :] >= -0.4)
    opti.subject_to(u[1, :] >= 0)
    opti.subject_to(u[1, :] <= 0.4)

    # solver settings
    opti.solver('ipopt')

    # return solver and variables
    return opti, x, u, x0, xf, points


""" --------------------------------------------------------------------------------------------------------------------
Koopman Motion Planner
-------------------------------------------------------------------------------------------------------------------- """

class KoopmanPlanner:

    def __init__(self, length, width, steps, horizon, nlidar, x_goal):
        """class constructor"""

        # store object properties
        self.length = length
        self.width = width
        self.steps = steps
        self.horizon = horizon
        self.nlidar = nlidar
        self.x_goal = x_goal

        # initialize Koopman model
        tmp = loadmat('/home/ccri-batch2-car3/f1tenth_ws/src/f1tenth_system/racecar/racecar/scripts/KoopmanModel.mat')

        self.A = tmp['A']
        self.B = tmp['B']
        self.D = tmp['D']
        self.b = tmp['u']
        self.w = tmp['w']

        # initialize optimizer
        self.solver, self.xSym, self.uSym, self.x0, self.xf, self.points = \
            init_optimizer(self.A, self.B, steps, horizon, nlidar, width, length)

    def plan(self, x, y, phi, v, lidar_data):
        """plan a trajectory by solving an optimal control problem"""

        # process lidar data
        points = process_lidar_data(lidar_data)
        ind = np.floor(np.linspace(0, points.shape[1]-1, self.nlidar))
        points = points[:, ind.astype(int)]

        # update goal state
        x_goal = transformation_local_frame(x, y, phi, self.x_goal)

        # initial state
        x0 = self.observable_map(np.array([0, 0, 0, v]))

        # set solver parameter
        self.solver.set_value(self.x0, x0)
        self.solver.set_value(self.points, points)
        self.solver.set_value(self.xf, x_goal)

        # solve optimal control problem
        sol = self.solver.solve()

        # get optimized values for variables
        u = sol.value(self.uSym)

        return u

    def observable_map(self, x):
        """map a state vector through the observable function"""

        tmp = np.sqrt(2 / self.D) * np.cos(x.T @ self.w.T + self.b)

        return np.vstack((np.expand_dims(x, axis=1), np.expand_dims(tmp[0], axis=1)))


""" --------------------------------------------------------------------------------------------------------------------
Publishers and Subscribers
-------------------------------------------------------------------------------------------------------------------- """

class PublisherSubscriber:
    """wrapper class that handles writing control commands and reading sensor measurements"""

    def __init__(self, controller, x0):
        """class constructor"""

        # publisher
        self.pub = rospy.Publisher("commands", Float32MultiArray, queue_size=1, latch=True)

        # subscribers
        self.sub_lidar = rospy.Subscriber("/scan", LaserScan, self.callback_lidar)
        self.sub_velocity = rospy.Subscriber("/vesc/odom/", Odometry, self.callback_velocity)
        self.sub_observer = rospy.Subscriber("observer", Float32MultiArray, self.callback_observer)

        # store motion planner and observer
        self.controller = controller

        # initialize control input and auxiliary variables
        self.goal_reached = False
        self.x_goal = controller.x_goal
        self.u = np.array([[0.0], [0.0]])
        self.x = x0[0]
        self.y = x0[1]
        self.theta = x0[2]

        # wait until first measurement is obtained
        rate = rospy.Rate(1000)

        while not hasattr(self, 'lidar_data') or not hasattr(self, 'velocity'):
            rate.sleep()

        # start timers for control command publishing and re-planning
        self.timer = rospy.Timer(rospy.Duration(0.01), self.callback_timer)
        rospy.spin()

    def callback_lidar(self, msg):
        """store lidar data"""

        self.lidar_data = np.asarray(msg.ranges)[0:1080]

    def callback_velocity(self, msg):
        """calculate absolute velocity from x- any y-components"""

        self.velocity = np.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)

    def callback_observer(self, msg):
        """store the current pose estimates"""

        self.x = msg.data[0]
        self.y = msg.data[1]
        self.theta = msg.data[2]

        if (self.x - self.x_goal[0])**2 + (self.y - self.x_goal[1])**2 < 0.5**2:
            self.goal_reached = True

    def callback_timer(self, timer):
        """obtain new control commands from the controller"""

        if not self.goal_reached:
            try:
                self.u = self.controller.plan(self.x, self.y, self.theta, self.velocity, self.lidar_data)
            except:
                self.u = self.u
        else:
            self.u = np.array([[0.0], [0.0]])

        # publish control commands   
        msg = Float32MultiArray()
        msg.data = [self.u[0, 0], self.u[1, 0]]

        self.pub.publish(msg) 


""" --------------------------------------------------------------------------------------------------------------------
Main Program
-------------------------------------------------------------------------------------------------------------------- """

if __name__ == '__main__':

    # initialize ROS
    rospy.init_node('autokoopman_online')

    # motion planner parameters
    steps = 4                                       # number of planning steps
    horizon = 10                                    # planning horizon (number of time steps during one planning step)
    nlidar = 20                                     # number of lidar beams
    x_goal = np.array([0.8, -3.8])                    # goal state
    x0 = np.array([0.0, -0.4, -0.5 * np.pi])        # initial state

    # parameter of the car
    length = 0.51
    width = 0.31

    # initialize motion planner
    planner = KoopmanPlanner(length, width, steps, horizon, nlidar, x_goal)

    # start control cycle
    PublisherSubscriber(planner, x0)
