import numpy as np
import cv2 as cv
import json
from manipulator import Manipulator
from grid import Grid
from planner import Planner
from math import floor
import matplotlib.pyplot as plt

f = open("config.json")
data = json.load(f)

rows = int(data["rows"])
cols = int(data["columns"])
pivot_x = float(data["pivot_x"])
pivot_y = float(data["pivot_y"])
l1 = float(data["l1"])
theta_1 = np.deg2rad(float(data["theta_1"]))
l2 = float(data["l2"])
theta_2 = np.deg2rad(float(data["theta_2"]))
obstacle_code = int(data["obstacle_code"])
obstacle_thickness = int(data["obstacle_thickness"])
manipulator_thickness = int(data["manipulator_thickness"])

def main():
    state_space = Grid(rows, cols)

    state_space.add_obstacles()

    manipulator = Manipulator(l1, theta_1, l2, theta_2, pivot_x, pivot_y)
    ee_start = manipulator.link_2_floored
    ((theta_1_a, theta_1_b), (theta_2_a, theta_2_b)) = manipulator.inverse_kinematic(ee_start)

    control_space = manipulator.create_control_space(state_space)

    planner = Planner(state_space, manipulator, control_space)
    print("-- Manipulator Positions --")
    print("Start : ", ee_start)
    goal = list(map(int, input("Set goal (space separated) : ").split()))
    # goal = [-165, 100]
    # goal = [ee_start[0] - 100, ee_start[1]]
    plan = planner.plan(goal)

    for i, control_output in enumerate(plan):
        
        manipulator_dynamic = Manipulator(l1, control_output[0], l2, control_output[1], pivot_x, pivot_y)

        dynamic_grid = Grid(rows, cols)
        dynamic_grid.add_obstacles()

        cv.line(dynamic_grid.grid, state_space.cartesian_to_viz(manipulator_dynamic.pivot_floored), state_space.cartesian_to_viz(manipulator_dynamic.link_1_floored), (255,0,0), manipulator_thickness)
        cv.line(dynamic_grid.grid, state_space.cartesian_to_viz(manipulator_dynamic.link_1_floored), state_space.cartesian_to_viz(manipulator_dynamic.link_2_floored), (255,0,0), manipulator_thickness)
        cv.circle(dynamic_grid.grid, state_space.cartesian_to_viz(manipulator_dynamic.pivot_floored), 8, (0, 0, 0), -1)
        cv.circle(dynamic_grid.grid, state_space.cartesian_to_viz(manipulator_dynamic.link_1_floored), 8, (0, 0, 0), -1)
        cv.circle(dynamic_grid.grid, state_space.cartesian_to_viz(manipulator_dynamic.link_2_floored), 8, (0, 0, 0), -1)
        cv.arrowedLine(dynamic_grid.grid, (cols // 2, rows // 2), (cols // 2, rows // 2 - 30), (0,255,0), 4)
        cv.arrowedLine(dynamic_grid.grid, (cols // 2, rows // 2), (cols // 2 + 30, rows // 2), (0,0,255), 4)
        cv.imshow("results/state", dynamic_grid.grid)
        cv.waitKey(1)

        if(i == 0):
            cv.imwrite("results/start_pos.png", dynamic_grid.grid)
        cv.imwrite("results/end_pos.png", dynamic_grid.grid)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

    theta_1s = []
    theta_2s = []

    for i in range(len(plan)):
        theta_1s.append(plan[i][0])
        theta_2s.append(plan[i][1])
    
    plt.plot(theta_1s, theta_2s)
    plt.title("theta_1 vs theta_2 plot")
    plt.xlabel("theta_1 (in radians)")
    plt.ylabel("theta_2 (in radians)")
    plt.savefig("theta_plot.png")
    plt.show()

if __name__=="__main__":
    main()