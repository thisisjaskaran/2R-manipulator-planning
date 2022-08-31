import numpy as np
import networkx as nx
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import floor
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

class Planner:
    def __init__(self, state_space, manipulator, control_space):
        self.state_space = state_space
        self.manipulator = manipulator
        self.control_space = control_space

        self.G = nx.grid_2d_graph(control_dim,control_dim)
        # print(G.nodes)
        # nx.set_edge_attributes(G,1,'weight')
        for i in tqdm(range(control_dim)):
            for j in range(control_dim):
                if(self.control_space.grid[i, j, 0] == obstacle_code):
                    self.G.remove_node((j,i))
        # G.remove_node((1,2))
        # pos = {(x,y):(y,-x) for x,y in G.nodes()}
        # nx.draw(G, pos=pos, 
        # node_color='lightgreen', 
        # with_labels=False,
        # node_size=600)
        # plt.show()

        self.start = self.manipulator.link_2_floored
        self.start_thetas_1, self.start_thetas_2 = self.manipulator.inverse_kinematic(self.start)
        self.start_control_1 = [self.start_thetas_1[0] * control_scaling, self.start_thetas_2[0] * control_scaling]
        self.start_control_2 = [self.start_thetas_1[1] * control_scaling, self.start_thetas_2[1] * control_scaling]
        
        self.goal = None
    
    def plan(self, goal):
        self.goal = goal
        self.goal_thetas_1, self.goal_thetas_2 = self.manipulator.inverse_kinematic(self.goal)
        self.goal_control_1 = [self.goal_thetas_1[0] * control_scaling, self.goal_thetas_2[0] * control_scaling]
        self.goal_control_2 = [self.goal_thetas_1[1] * control_scaling, self.goal_thetas_2[1] * control_scaling]

        # print(self.start_thetas_1, " -> ", self.goal_thetas_1)
        # print(self.start_thetas_2, " -> ", self.goal_thetas_2)

        # print(self.control_space.grid[:, :, 0]/255)
        
        # Case 1
        start_1 = self.control_space.cartesian_to_viz(self.start_control_1)
        goal_1 = self.control_space.cartesian_to_viz(self.goal_control_1)

        # Case 2
        start_2 = self.control_space.cartesian_to_viz(self.start_control_2)
        goal_2 = self.control_space.cartesian_to_viz(self.goal_control_2)

        start_u, start_v = floor(start_1[0]), floor(start_1[1])
        goal_u, goal_v = floor(goal_1[0]), floor(goal_1[1])

        path = nx.astar_path(self.G, (start_u, start_v), (goal_u, goal_v))
        control_outputs = []
        for point in path:
            theta_1 = point[0] / control_scaling
            theta_2 = point[1] / control_scaling

            pos = self.manipulator.forward_kinematics((theta_1, theta_2))
            control_outputs.append([theta_1, theta_2])

        return control_outputs