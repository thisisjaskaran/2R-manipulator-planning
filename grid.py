import numpy as np
import json
import cv2 as cv
from math import sin, cos

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

class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        self.grid = np.ones((self.rows, self.cols, 3)).astype(np.uint8) * 255
    
    def add_obstacles(self):
        t1 = (230, 310)
        t2 = (160, 310)
        t3 = (185, 365)
        
        cv.line(self.grid, t1, t2, (obstacle_code, obstacle_code, obstacle_code), obstacle_thickness)
        cv.line(self.grid, t2, t3, (obstacle_code, obstacle_code, obstacle_code), obstacle_thickness)
        cv.line(self.grid, t1, t3, (obstacle_code, obstacle_code, obstacle_code), obstacle_thickness)

        center = (240,365)

        cv.circle(self.grid, center, 27, (obstacle_code, obstacle_code, obstacle_code), obstacle_thickness)

        l1 = (370,405)
        l2 = (440,415)

        cv.line(self.grid, l1, l2, (obstacle_code, obstacle_code, obstacle_code), obstacle_thickness)

        cv.imwrite("results/grid.png", self.grid)
    
    def cartesian_to_viz(self, point):
        x = point[0]
        y = point[1]

        u = x + self.cols // 2 - 1
        v = self.rows // 2 - y - 1

        return (u,v)
    
    def viz_to_cartesian(self, point):
        u = point[0]
        v = point[1]

        x = u - self.cols // 2 + 1
        y = self.rows // 2 - v - 1

        return (x,y)