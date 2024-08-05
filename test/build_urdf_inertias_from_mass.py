"""Inertia computations for Franka end effector, given each link's mass and
geometric properties.  Assumes each link's mass is evenly distributed."""

import pdb
import torch
from torch import Tensor

import dair_pll.inertia


base = 'base'
link = 'link'
tip = 'tip'

masses =  {base: 0.0779312, link: 0.1340688, tip: 0.057}
radii =   {base: 0.0315,    link: 0.0127,    tip: 0.0195}
heights = {base: 0.0096,    link: 0.1016}

def compute_cylinder_inertias(mass, rad, height):
    Ixx = (1/12) * mass * (3*rad**2 + height**2)
    Izz = (1/2) * mass * rad**2

    print(f'Ixx = Iyy: {Ixx} \nIzz: {Izz}\n')

def compute_sphere_inertias(mass, rad):
    Ixx = (2/5) * mass * rad**2
    print(f'Ixx: {Ixx}\n')


print(f'Base:')
compute_cylinder_inertias(masses[base], radii[base], heights[base])

print(f'Link:')
compute_cylinder_inertias(masses[link], radii[link], heights[link])

print(f'Tip:')
compute_sphere_inertias(masses[tip], radii[tip])


# Sanity check a value from the drake_pytorch generated function against values
# in the Franka URDF.
robot_panda_link1_m = 2.74
robot_panda_link1_com = [0, -0.0324958, -0.0675818]
robot_panda_link1_Ixx = 0.0180416958283
robot_panda_link1_Ixy=0.0
robot_panda_link1_Ixz=0.0
robot_panda_link1_Iyy=0.0159136071891
robot_panda_link1_Iyz=0.0046758424612
robot_panda_link1_Izz=0.0062069082712

# (robot_panda_link1_m * ((robot_panda_link1_I(2) / robot_panda_link1_m) + pow(robot_panda_link1_com(0), 2) + pow(robot_panda_link1_com(1), 2)))
val = robot_panda_link1_m * (
    (robot_panda_link1_Izz / robot_panda_link1_m) + \
    (robot_panda_link1_com[0]**2) + (robot_panda_link1_com[1]**2))


# Compute "ground-truth" cube inertia values.
mass = 0.37
half_length = 0.0524

side_mass = mass / 6

Ixx = (1/12) * side_mass * half_length**2
Iyy = (1/12) * side_mass * half_length**2
Izz = (1/12) * side_mass * (2*half_length)**2

inertia_tensor_cube = torch.zeros((3,3))

# Do the tops and bottom (z-axis).
top_I_BBcm_B = Tensor([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
top_p_BcmBcube_B = Tensor([0, 0, half_length])

top_I_BBcube_B = dair_pll.inertia.parallel_axis_theorem(
    top_I_BBcm_B, Tensor([side_mass]), top_p_BcmBcube_B)

# Do the left and right (x-axis).
right_I_BBcm_B = Tensor([[Izz, 0, 0], [0, Iyy, 0], [0, 0, Ixx]])
right_p_BcmBcube_B = Tensor([half_length, 0, 0])

right_I_BBcube_B = dair_pll.inertia.parallel_axis_theorem(
    right_I_BBcm_B, Tensor([side_mass]), right_p_BcmBcube_B)

# Do the front and back (y-axis).
front_I_BBcm_B = Tensor([[Ixx, 0, 0], [0, Izz, 0], [0, 0, Iyy]])
front_p_BcmBcube_B = Tensor([0, half_length, 0])

front_I_BBcube_B = dair_pll.inertia.parallel_axis_theorem(
    front_I_BBcm_B, Tensor([side_mass]), front_p_BcmBcube_B)

# Combine.
inertia_tensor_cube = 2*top_I_BBcube_B + 2*right_I_BBcube_B + 2*front_I_BBcube_B
inertia_tensor_cube = inertia_tensor_cube.reshape(3,3)

print('Cube inertia tensor:\n', inertia_tensor_cube)
print('Element 0,0:\n', inertia_tensor_cube[0,0].item())

pdb.set_trace()
