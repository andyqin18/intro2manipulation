import numpy as np
import random
from assignment_7_helper import World
from scipy.spatial.transform import Rotation as R
import time
import pybullet as p


np.set_printoptions(precision=3, suppress=True)

def move(env, target, angle, is_open, step_size):
    '''Move the robot to given state'''
    # Get target state
    rob_state = env.get_robot_state()
    rob_state[3] = normalize_angle(rob_state[3])
    target_state = np.zeros(4)
    target_state[:3] = target
    target_state[3] = np.radians(angle) 

    # Get steps of robot command
    diff = target_state - rob_state
    step_num = int(np.max(np.abs(diff / step_size)))
    steps = np.linspace(rob_state, target_state, step_num)
    robot_command = []
    for i in range(step_num):
        command = np.zeros(5)
        command[:4] = steps[i, :]
        command[-1] = 0.2 if is_open else 0
        robot_command.append(command)

    env.robot_command(robot_command)
    print(np.array(robot_command))

def normalize_angle(angle):
    '''Clip angle into -pi to pi'''
    # Normalize to [0, 2pi)
    angle = angle % (2 * np.pi)
    # Shift to [-pi, pi]
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle

def switch(env, open):
    '''Slowly switch gripper state'''
    robot_state = env.get_robot_state()
    robot_state[3] = normalize_angle(robot_state[3])

    command = np.zeros((5,5))
    command[:,:4] = np.tile(robot_state, (5,1))
    start = 0 if open else 0.2
    end = 0.2 if open else 0
    command[:,-1] = np.linspace(start, end, 5)
    env.robot_command([np.array(row) for row in command])
    print(command)

def pick2drop(env, obj_id):
    '''Pick and place object to (0, 0, 0)'''
    # Get object state
    obj_state = env.get_obj_state()
    obj_loc = obj_state[obj_id, :3]
    obj_quat = obj_state[obj_id, 3:]
    obj_rot = R.from_quat(obj_quat).as_matrix()

    # Get orientation of object
    theta_z = np.arctan2(obj_rot[1, 0], obj_rot[0, 0])
    theta_z = np.degrees(theta_z)

    # Set target loc
    target = obj_loc
    target_up = obj_loc.copy()
    target_up[2] += 0.15
    target[2] = 0.002

    # Hover - Dive - Grasp - Hover - Rotate - Move - Drop
    move(env, target_up, angle=theta_z, is_open=True, step_size=0.07)
    move(env, target, angle=theta_z, is_open=True, step_size=0.02)
    switch(env, open=False)
    move(env, target_up, angle=theta_z, is_open=False, step_size=0.07)
    move(env, target_up, angle=0, is_open=False, step_size=0.07)
    move(env, np.array([0., 0., target_up[2]]), angle=0, is_open=False, step_size=0.07)
    time.sleep(1)
    switch(env, open=True)
    
    

def stack():
    """
    function to stack objects
    :return: average height of objects
    """
    # DO NOT CHANGE: initialize world
    env = World()

    # ============================================
    # YOUR CODE HERE:
    time.sleep(30)
    
    # Initialize
    robot_command = [np.array([0., 0.,    0.025, 0, 0.])]
    env.robot_command(robot_command)

    # Push to separate
    waypoints = np.array([[-0.05,   0.1,    0.025],
                          [0.2,     0.4,    0.025],
                          [0.2,     0.4,    0.1]])
    
    for i in range(waypoints.shape[0]):
        move(env, target=waypoints[i, :], angle=0, is_open=False, step_size=0.02)
    time.sleep(3)
    
    # Pick and place each object
    obj_state = env.get_obj_state()
    for i in range(obj_state.shape[0]):
        pick2drop(env, obj_id=i)
  
    # ============================================
    # DO NOT CHANGE: getting average height of objects:
    obj_state = env.get_obj_state()
    avg_height = np.mean(obj_state[:, 2])
    print("Average Object Height: {:4.3f}".format(avg_height))
    return env, avg_height

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    stack()
