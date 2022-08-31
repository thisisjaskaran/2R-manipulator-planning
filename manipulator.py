import numpy as np
import json
from math import sin, cos, floor
import math
from grid import Grid
import cv2 as cv

f = open("config.json")
data = json.load(f)

rows = int(data["rows"])
cols = int(data["columns"])
pivot_x = int(data["pivot_x"])
pivot_y = int(data["pivot_y"])
l1 = float(data["l1"])
theta_1 = np.deg2rad(float(data["theta_1"]))
l2 = float(data["l2"])
theta_2 = np.deg2rad(float(data["theta_2"]))
obstacle_code = int(data["obstacle_code"])
obstacle_thickness = int(data["obstacle_thickness"])
manipulator_thickness = int(data["manipulator_thickness"])
control_dim = int(data["control_dim"])
control_scaling = int(data["control_scaling"])

class Manipulator:
    def __init__(self, l1, theta_1, l2, theta_2, pivot_x, pivot_y):
        self.pivot = (pivot_x, pivot_y)
        self.pivot_floored = (floor(pivot_x), floor(pivot_y))

        self.l1 = l1
        self.theta_1 = theta_1
        self.l2 = l2
        self.theta_2 = theta_2

        self.link_1 = ( pivot_x + l1 * cos(self.theta_1),
                        pivot_y + l1 * sin(self.theta_1))
        self.link_2 = ( self.link_1[0] + l2 * cos(self.theta_1 + self.theta_2),
                        self.link_1[1] + l2 * sin(self.theta_1 + self.theta_2))

        self.link_1_floored = ( floor(pivot_x + l1 * cos(self.theta_1)),
                                floor(pivot_y + l1 * sin(self.theta_1)))
        self.link_2_floored = ( floor(self.link_1[0] + l2 * cos(self.theta_1 + self.theta_2)),
                                floor(self.link_1[1] + l2 * sin(self.theta_1 + self.theta_2)))
    
    def forward_kinematics(self, point):
        theta_1 = point[0]
        theta_2 = point[1]

        x = self.l1 * cos(theta_1) + self.l2 * cos(theta_1 + theta_2)
        y = self.l1 * sin(theta_1) + self.l2 * sin(theta_1 + theta_2)

        return (x,y)
        
    def inverse_kinematic(self, point):
        x = point[0]
        y = point[1]
        
        if(abs((x**2 + y**2 - self.l1**2 - self.l2**2) / (2*self.l1*self.l2)) > 1.0):
            return (None, None), (None, None)

        theta_2_a = np.arccos((x**2 + y**2 - self.l1**2 - self.l2**2) / (2*self.l1*self.l2))
        theta_2_b = 2 * np.pi - theta_2_a

        theta_1_a = math.atan2(y,x) - math.atan2(l2 * sin(theta_2_a), l1 + l2 * sin(theta_2_a))
        theta_1_b = math.atan2(y,x) - math.atan2(l2 * sin(theta_2_b), l1 + l2 * sin(theta_2_b))

        return (theta_1_a, theta_1_b), (theta_2_a, theta_2_b)
    
    def create_control_space(self, Grid_):
        grid = Grid_.grid

        control_space = Grid(control_dim,control_dim)

        for i in range(rows):

            for j in range(cols):

                if(grid[i, j, 0] == obstacle_code):
                    thetas_1, thetas_2 = self.inverse_kinematic(Grid_.viz_to_cartesian((i,j)))

                    if(thetas_1[0] is None or thetas_1[1] is None or thetas_2[0] is None or thetas_2[1] is None):
                        continue

                    theta_1_a = thetas_1[0] * control_scaling
                    theta_1_b = thetas_1[1] * control_scaling
                    theta_2_a = thetas_2[0] * control_scaling
                    theta_2_b = thetas_2[1] * control_scaling

                    u_a, v_a = floor(theta_1_a + control_dim // 2), floor(control_dim // 2 - theta_2_a - 1)
                    u_b, v_b = floor(theta_1_b + control_dim // 2), floor(control_dim // 2 - theta_2_b - 1)

                    control_space.grid[u_a, v_a] = [0, 0, 0]
                    control_space.grid[u_b, v_b] = [0, 0, 0]
        
        cv.arrowedLine(control_space.grid, (control_dim // 2, control_dim // 2), (control_dim // 2, control_dim // 2 - 30), (0,255,0), 4)
        cv.arrowedLine(control_space.grid, (control_dim // 2, control_dim // 2), (control_dim // 2 + 30, control_dim // 2), (0,0,255), 4)
        cv.imwrite("results/control.png", control_space.grid)
        return control_space