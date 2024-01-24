import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import yaml
import sys
import torch
from scipy.spatial.transform import Rotation as R
import pydrake
from pydrake.all import StartMeshcat, RandomGenerator, BodyIndex, Parser
from pydrake.common import FindResourceOrThrow
from pydrake.common.eigen_geometry import Quaternion, AngleAxis, Isometry3
from pydrake.geometry import (
    Box,
    HalfSpace,
    SceneGraph,
    Sphere,
)
from pydrake.math import (RollPitchYaw, RotationMatrix, RigidTransform)
from pydrake.multibody.tree import (
    PrismaticJoint,
    SpatialInertia,
    UniformGravityFieldElement,
    UnitInertia, 
    world_model_instance
)
from pydrake.multibody.math import SpatialVelocity
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    MultibodyPlant
)

from pydrake.forwarddiff import gradient
from pydrake.multibody.parsing import Parser
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import MeshcatVisualizer, MeshcatVisualizerParams

"""Simulate a cube toss using the trajectory from the learned model.
"""

urdf_file = "../assets/contactnets_cube.urdf"
def AddShape(plant, shape, name, mass=1, mu=1, com=np.array([0.0, 0.0, 0.0]), inertia=UnitInertia(), color=[0.5, 0.5, 0.9, 1.0]):
    instance = plant.AddModelInstance(name)
    body = plant.AddRigidBody(
        name,
        instance,
        SpatialInertia(
            mass=mass, p_PScm_E=com, G_SP_E=inertia
        ),
    )
    if plant.geometry_source_is_registered():
        if isinstance(shape, Box):
            plant.RegisterCollisionGeometry(
                body,
                RigidTransform(),
                Box(
                    shape.width() - 0.001,
                    shape.depth() - 0.001,
                    shape.height() - 0.001,
                ),
                name,
                CoulombFriction(mu, mu),
            )
            i = 0
            for x in [-shape.width() / 2.0, shape.width() / 2.0]:
                for y in [-shape.depth() / 2.0, shape.depth() / 2.0]:
                    for z in [-shape.height() / 2.0, shape.height() / 2.0]:
                        plant.RegisterCollisionGeometry(
                            body,
                            RigidTransform([x, y, z]),
                            Sphere(radius=1e-7),
                            f"contact_sphere{i}",
                            CoulombFriction(mu, mu),
                        )
                        i += 1
        else:
            plant.RegisterCollisionGeometry(
                body, RigidTransform(), shape, name, CoulombFriction(mu, mu)
            )

        plant.RegisterVisualGeometry(
            body, RigidTransform(), shape, name, color
        )

    return instance

def AddGround(plant):
    ground_instance = plant.AddModelInstance("ground")
    world_body = plant.world_body()
    ground_shape = Box(10., 10., 10.)
    ground_body = plant.AddRigidBody("ground", ground_instance, SpatialInertia(
        mass=10.0, p_PScm_E=np.array([0., 0., 0.]),
        G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
    plant.WeldFrames(world_body.body_frame(), ground_body.body_frame(),
                    RigidTransform(Isometry3(rotation=np.eye(3), translation=[0, 0, -5])))
    plant.RegisterVisualGeometry(
        ground_body, RigidTransform.Identity(), ground_shape, "ground_vis",
        np.array([0.5, 0.5, 0.5, 1.]))
    plant.RegisterCollisionGeometry(
        ground_body, RigidTransform.Identity(), ground_shape, "ground_col",
        CoulombFriction(0.9, 0.8))

def simulate_cube_toss(params, trajectory_dir):
    """Simulate based on physical params

    Args:
        params (Dict): _description_
        trajectory_dir (str): _description_
    """
    if params == None:
        cube_body_m = 0.37
        cube_body_com_x = 0
        cube_body_com_y = 0
        cube_body_com_z = 0
        cube_body_Ixx = 0.00081
        cube_body_Iyy = 0.00081
        cube_body_Izz = 0.00081
        cube_body_Ixy = 0.0
        cube_body_Ixz = 0.0
        cube_body_Iyz = 0.0
        cube_body_mu = 0.15
        cube_body_len_x = 0.1048
        cube_body_len_y = 0.1048
        cube_body_len_z = 0.1048
    else:
        cube_body_m = params['cube_body_m']
        cube_body_com_x = params['cube_body_com_x']
        cube_body_com_y = params['cube_body_com_y']
        cube_body_com_z = params['cube_body_com_z']
        cube_body_Ixx = params['cube_body_I_xx']
        cube_body_Iyy = params['cube_body_I_yy']
        cube_body_Izz = params['cube_body_I_zz']
        cube_body_Ixy = params['cube_body_I_xy']
        cube_body_Ixz = params['cube_body_I_xz']
        cube_body_Iyz = params['cube_body_I_yz']
        cube_body_mu = params['cube_body_mu']
        cube_body_len_x = params['cube_body_len_x']
        cube_body_len_y = params['cube_body_len_y']
        cube_body_len_z = params['cube_body_len_z']

    # np.random.seed(42)
    # random.seed(42)
    # rng = np.random.default_rng(135)  # this is for python
    # generator = RandomGenerator(rng.integers(0, 1000))
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=0.0001))
    shape = Box(cube_body_len_x, cube_body_len_y, cube_body_len_z)
    name = 'cube'
    com = np.array([cube_body_com_x, cube_body_com_y, cube_body_com_z])
    inertia = UnitInertia(cube_body_Ixx, cube_body_Iyy, cube_body_Izz, cube_body_Ixy, cube_body_Ixz, cube_body_Iyz)
    AddShape(plant, shape, name, mass=cube_body_m, mu=cube_body_mu, com=com, inertia=inertia, color=[0.5, 0.5, 0.9, 1.0])
    AddGround(plant)
    plant.Finalize()
    
    meshcat = StartMeshcat()
    params = MeshcatVisualizerParams()
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
    
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    sg_context = diagram.GetMutableSubsystemContext(scene_graph, diagram_context)
    q0 = plant.GetPositions(plant_context).copy()
    v0 = plant.GetVelocities(plant_context).copy()
    plant.SetPositions(plant_context, q0)
    plant.SetVelocities(plant_context, v0)
    traj = torch.load(trajectory_dir)[0, :] #q_t(wxyz), p_t, w_t, dp_t
    print(f'traj: {traj.size()}')
    # print(type(traj[4:7]))
    # print(traj[4:7].shape)
    p_t = traj[:3]
    q_t = traj[3:7]
    dp_t = traj[7:10]
    w_t = traj[10:]
    print(q_t.shape, p_t.shape, dp_t.shape, w_t.shape)
    rot = RotationMatrix(R.from_quat(traj[3:7]).as_matrix())
    pose = RigidTransform(rot, traj[:3].numpy())
    vel = SpatialVelocity(traj[10:13].numpy(), traj[7:10].numpy())
    for body_index in plant.GetFloatingBaseBodies():
        plant.SetFreeBodyPose(plant_context, plant.get_body(body_index), pose)
        plant.SetFreeBodySpatialVelocity(plant.get_body(body_index), vel, plant_context)
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)
    q0_final = plant.GetPositions(plant_context).copy()
    print(q0, q0_final)
    while True:
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        cube_position = plant.EvalBodyPoseInWorld(plant_context, plant.get_body(BodyIndex(0))).translation()
        print("Reinitializing...")
        if cube_position[2] <= 0.01:  
            # The cube has landed, so reset the simulation
            simulator.get_mutable_context().SetTime(0.)
            plant.SetPositions(plant_context, q0)
            plant.SetVelocities(plant_context, v0)
            for body_index in plant.GetFloatingBaseBodies():
                plant.SetFreeBodyPose(plant_context, plant.get_body(body_index), pose)
                plant.SetFreeBodySpatialVelocity(plant.get_body(body_index), vel, plant_context)
            simulator.Initialize()
    
def simulate_cube_toss_with_traj(params, trajectory_dir):
    """Simulate a trajectory

    Args:
        params (Dict): _description_
        trajectory_dir (str): _description_
    """
    print(f'Reading traj from {trajectory_dir}')
    if params == None:
        cube_body_m = 0.37
        cube_body_com_x = 0
        cube_body_com_y = 0
        cube_body_com_z = 0
        cube_body_Ixx = 0.00081
        cube_body_Iyy = 0.00081
        cube_body_Izz = 0.00081
        cube_body_Ixy = 0.0
        cube_body_Ixz = 0.0
        cube_body_Iyz = 0.0
        cube_body_mu = 0.15
        cube_body_len_x = 0.1048
        cube_body_len_y = 0.1048
        cube_body_len_z = 0.1048
    else:
        cube_body_m = params['cube_body_m']
        cube_body_com_x = params['cube_body_com_x']
        cube_body_com_y = params['cube_body_com_y']
        cube_body_com_z = params['cube_body_com_z']
        cube_body_Ixx = params['cube_body_I_xx']
        cube_body_Iyy = params['cube_body_I_yy']
        cube_body_Izz = params['cube_body_I_zz']
        cube_body_Ixy = params['cube_body_I_xy']
        cube_body_Ixz = params['cube_body_I_xz']
        cube_body_Iyz = params['cube_body_I_yz']
        cube_body_mu = params['cube_body_mu']
        cube_body_len_x = params['cube_body_len_x']
        cube_body_len_y = params['cube_body_len_y']
        cube_body_len_z = params['cube_body_len_z']
    traj = torch.load(trajectory_dir) #q_t(wxyz), p_t, w_t, dp_t
    print(f'traj loaded: {traj.size()}') #N,13
    p_t = traj[:,4:7].numpy() #N,3
    q_t = traj[:,:4].numpy() #N,4, w,x,y,z
    q_t_shuffled = np.concatenate((q_t[:, 1:], q_t[:, 0].reshape(-1,1)), axis=1) ##N,4, x,y,z,w
    dp_t = traj[:,10:].numpy() #N,3
    w_t = traj[:,7:10].numpy() #N,3, in body frame
    w_t_world = np.zeros_like(w_t)
    for i in range(traj.shape[0]):
        rot = R.from_quat(q_t_shuffled[i]).as_matrix()
        w_t_i = w_t[i]
        w_t_world[i] = rot @ w_t_i.T
    # w_t_world = R.from_quat(q_t_shuffled).as_matrix() @ w_t.T
    print(p_t.shape, q_t_shuffled.shape, dp_t.shape, w_t_world.shape)
    
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=0.0001))
    shape = Box(cube_body_len_x, cube_body_len_y, cube_body_len_z)
    name = 'cube'
    com = np.array([cube_body_com_x, cube_body_com_y, cube_body_com_z])
    inertia = UnitInertia(cube_body_Ixx, cube_body_Iyy, cube_body_Izz, cube_body_Ixy, cube_body_Ixz, cube_body_Iyz)
    AddShape(plant, shape, name, mass=cube_body_m, mu=cube_body_mu, com=com, inertia=inertia, color=[0.5, 0.5, 0.9, 1.0])
    # AddGround(plant)
    # add ground at z=0
    halfspace_transform = RigidTransform()
    friction = CoulombFriction(1.0, 1.0)
    plant.RegisterCollisionGeometry(plant.world_body(), halfspace_transform,
                                    HalfSpace(), "ground", friction)
    plant.Finalize()
    
    meshcat = StartMeshcat()
    params = MeshcatVisualizerParams()
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
    
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    sg_context = diagram.GetMutableSubsystemContext(scene_graph, diagram_context)
    q0 = plant.GetPositions(plant_context).copy()
    v0 = plant.GetVelocities(plant_context).copy()
    plant.SetPositions(plant_context, q0)
    plant.SetVelocities(plant_context, v0)
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)
    # Set initial system states
    p_t_0 = p_t[0]
    q_t_0 = q_t_shuffled[0]
    dp_t_0 = dp_t[0]
    w_t_0 = w_t_world[0]
    rot_0 = RotationMatrix(R.from_quat(q_t_0).as_matrix())
    pose_0 = RigidTransform(rot_0, p_t_0)
    vel_0 = SpatialVelocity(w_t_0, dp_t_0)
    for body_index in plant.GetFloatingBaseBodies():
        plant.SetFreeBodyPose(plant_context, plant.get_body(body_index), pose_0)
        plant.SetFreeBodySpatialVelocity(plant.get_body(body_index), vel_0, plant_context)
    simulator_time = 0.0
    time_step = 0.01
    trajectory_length = len(q_t)
    playback_speed = 1.0
    while True:
        for i in range(1, trajectory_length):
            print(f'z position: {p_t[i, -1]}')
            target_time = i * time_step
            if simulator_time < target_time:
                simulator.AdvanceTo(target_time)
                simulator_time = simulator.get_context().get_time()
            q_t_i = q_t_shuffled[i]
            p_t_i = p_t[i]
            dp_t_i = dp_t[i]
            w_t_i = w_t_world[i]
            rot_i = RotationMatrix(R.from_quat(q_t_i).as_matrix())
            pose_i = RigidTransform(rot_i, p_t_i)
            vel_i = SpatialVelocity(w_t_i, dp_t_i)
            for body_index in plant.GetFloatingBaseBodies():
                plant.SetFreeBodyPose(plant_context, plant.get_body(body_index), pose_i)
                plant.SetFreeBodySpatialVelocity(plant.get_body(body_index), vel_i, plant_context)
            time.sleep(time_step / playback_speed)
        print("Reinitializing...")
        simulator_time = 0.0
        simulator.get_mutable_context().SetTime(0.)
        plant.SetPositions(plant_context, q0)
        plant.SetVelocities(plant_context, v0)
        p_t_0 = p_t[0]
        q_t_0 = q_t_shuffled[0]
        dp_t_0 = dp_t[0]
        w_t_0 = w_t_world[0]
        rot_0 = RotationMatrix(R.from_quat(q_t_0).as_matrix())
        pose_0 = RigidTransform(rot_0, p_t_0)
        vel_0 = SpatialVelocity(w_t_0, dp_t_0)
        for body_index in plant.GetFloatingBaseBodies():
            plant.SetFreeBodyPose(plant_context, plant.get_body(body_index), pose_0)
            plant.SetFreeBodySpatialVelocity(plant.get_body(body_index), vel_0, plant_context)
        simulator.Initialize()
            
def simulate_cube_and_franka(trajectory_dir):
    
    traj = torch.load(trajectory_dir) #q_t(wxyz), p_t, w_t, dp_t
    print(f'traj loaded: {traj.size()}') #N,13
    p_t = traj[:,4:7].numpy() #N,3
    q_t = traj[:,:4].numpy() #N,4, w,x,y,z
    # q_t_shuffled = np.concatenate((q_t[:, 1:], q_t[:, 0].reshape(-1,1)), axis=1) ##N,4, x,y,z,w
    dp_t = traj[:,10:].numpy() #N,3
    w_t = traj[:,7:10].numpy() #N,3, in body frame
    w_t_world = np.zeros_like(w_t)
    for i in range(traj.shape[0]):
        rot = R.from_quat(q_t[i]).as_matrix()
        w_t_i = w_t[i]
        w_t_world[i] = rot @ w_t_i.T
    # w_t_world = R.from_quat(q_t_shuffled).as_matrix() @ w_t.T
    print(p_t.shape, q_t.shape, dp_t.shape, w_t_world.shape)
    traj = np.concatenate((q_t, p_t, w_t, dp_t), axis=1) #N,13
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    X_model = RigidTransform.Identity()
    parser = Parser(plant)
    model_file = FindResourceOrThrow(
        "drake/manipulation/models/franka_description/urdf/panda_arm_hand_wide_finger.urdf"
    )
    cube_model_file = FindResourceOrThrow("drake/../../../../../dair_pll_latest/assets/contactnets_cube_sim.urdf")
    model = parser.AddModelFromFile(model_file)
    cube_model = parser.AddModelFromFile(cube_model_file)
    
    # plant.WeldFrames(
    #     plant.world_frame(),
    #     plant.GetFrameByName("body", cube_model),
    #     X_model,
    # )

    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("panda_link0", model),
        X_model,
    )
    plant.Finalize()
    params = MeshcatVisualizerParams()
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, params
    )
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    q0 = plant.GetPositions(plant_context).copy()
    v0 = plant.GetVelocities(plant_context).copy()
    plant.get_actuation_input_port().FixValue(plant_context, np.zeros(9))
    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)
    simulator_time = 0.0
    time_step = 0.01
    trajectory_length = len(q_t)
    playback_speed = 1.0
    while True:
        for i in range(1, trajectory_length):
            target_time = i * time_step
            if simulator_time < target_time:
                simulator.AdvanceTo(target_time)
                simulator_time = simulator.get_context().get_time()
            
            q_drake = traj[i, :7]
            v_drake = traj[i, 7:]
            print(plant.num_positions(cube_model))
            print(plant.num_velocities(cube_model))
            print(q_drake.shape, v_drake.shape)
            plant.SetPositions(plant_context, cube_model, q_drake)
            plant.SetVelocities(plant_context, cube_model, v_drake)
            time.sleep(time_step / playback_speed)
        print("Reinitializing...")
        simulator_time = 0.0
        simulator.get_mutable_context().SetTime(0.)
        plant.SetPositions(plant_context, q0)
        plant.SetVelocities(plant_context, v0)
        q_drake = traj[0, :7]
        v_drake = traj[0, 7:]
        plant.SetPositions(plant_context, cube_model, q_drake)
        plant.SetVelocities(plant_context, cube_model, v_drake)
        simulator.Initialize()

def simulate_toss_with_traj(trajectory_dir, urdf_dir):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=0.0001))
    X_model = RigidTransform.Identity()
    parser_ = Parser(plant)
    model_file = FindResourceOrThrow(urdf_dir)
    model = parser_.AddModelFromFile(model_file)
    plant.Finalize()

    traj = torch.load(trajectory_dir) #q_t(wxyz), p_t, w_t, dp_t
    print(f'traj loaded: {traj.size()}') #N,13
    p_t = traj[:,4:7].numpy() #N,3
    q_t = traj[:,:4].numpy() #N,4, w,x,y,z
    q_t_shuffled = np.concatenate((q_t[:, 1:], q_t[:, 0].reshape(-1,1)), axis=1) ##N,4, x,y,z,w
    dp_t = traj[:,10:].numpy() #N,3
    w_t = traj[:,7:10].numpy() #N,3, in body frame
    w_t_world = np.zeros_like(w_t)
    for i in range(traj.shape[0]):
        rot = R.from_quat(q_t_shuffled[i]).as_matrix()
        w_t_i = w_t[i]
        w_t_world[i] = rot @ w_t_i.T
    # w_t_world = R.from_quat(q_t_shuffled).as_matrix() @ w_t.T
    print(p_t.shape, q_t_shuffled.shape, dp_t.shape, w_t_world.shape)
    
    meshcat = StartMeshcat()
    params = MeshcatVisualizerParams()
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    sg_context = diagram.GetMutableSubsystemContext(scene_graph, diagram_context)
    q0 = plant.GetPositions(plant_context).copy()
    v0 = plant.GetVelocities(plant_context).copy()
    plant.SetPositions(plant_context, q0)
    plant.SetVelocities(plant_context, v0)
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(True)
    # Set initial system states
    p_t_0 = p_t[0]
    q_t_0 = q_t_shuffled[0]
    dp_t_0 = dp_t[0]
    w_t_0 = w_t_world[0]
    rot_0 = RotationMatrix(R.from_quat(q_t_0).as_matrix())
    pose_0 = RigidTransform(rot_0, p_t_0)
    vel_0 = SpatialVelocity(w_t_0, dp_t_0)
    for body_index in plant.GetFloatingBaseBodies():
        plant.SetFreeBodyPose(plant_context, plant.get_body(body_index), pose_0)
        plant.SetFreeBodySpatialVelocity(plant.get_body(body_index), vel_0, plant_context)
    simulator_time = 0.0
    time_step = 0.01
    trajectory_length = len(q_t)
    playback_speed = 1.0
    while True:
        initial_time = simulator.get_context().get_time()
        for i in range(1, trajectory_length):
            print(f'z position: {p_t[i, -1]}')
            q_t_i = q_t_shuffled[i]
            p_t_i = p_t[i]
            dp_t_i = dp_t[i]
            w_t_i = w_t_world[i]
            rot_i = RotationMatrix(R.from_quat(q_t_i).as_matrix())
            pose_i = RigidTransform(rot_i, p_t_i)
            vel_i = SpatialVelocity(w_t_i, dp_t_i)
            for body_index in plant.GetFloatingBaseBodies():
                plant.SetFreeBodyPose(plant_context, plant.get_body(body_index), pose_i)
                plant.SetFreeBodySpatialVelocity(plant.get_body(body_index), vel_i, plant_context)
            target_time = initial_time + i * time_step
            simulator.AdvanceTo(target_time)
            time.sleep(time_step / playback_speed)
        simulator.get_mutable_context().SetTime(initial_time)
        print("Reinitializing...")
        simulator_time = 0.0
        simulator.get_mutable_context().SetTime(0.)
        plant.SetPositions(plant_context, q0)
        plant.SetVelocities(plant_context, v0)
        p_t_0 = p_t[0]
        q_t_0 = q_t_shuffled[0]
        dp_t_0 = dp_t[0]
        w_t_0 = w_t_world[0]
        rot_0 = RotationMatrix(R.from_quat(q_t_0).as_matrix())
        pose_0 = RigidTransform(rot_0, p_t_0)
        vel_0 = SpatialVelocity(w_t_0, dp_t_0)
        for body_index in plant.GetFloatingBaseBodies():
            plant.SetFreeBodyPose(plant_context, plant.get_body(body_index), pose_0)
            plant.SetFreeBodySpatialVelocity(plant.get_body(body_index), vel_0, plant_context)
        simulator.Initialize()

def postprocess(folder_path, tosses_to_remove):
    tosses_to_remove.sort(reverse=True)
    for toss in tosses_to_remove:
        filename_to_remove = f"{toss - 1}.pt"
        full_path_to_remove = os.path.join(folder_path, filename_to_remove)
        if not os.path.exists(full_path_to_remove):
            print(f"File corresponding to toss {toss} does not exist!")
            continue
        os.remove(full_path_to_remove)
        number_to_remove = toss - 1
        current_number = number_to_remove + 1
        while True:
            current_filename = f"{current_number}.pt"
            full_current_path = os.path.join(folder_path, current_filename)
            if os.path.exists(full_current_path):
                new_filename = f"{current_number - 1}.pt"
                full_new_path = os.path.join(folder_path, new_filename)
                os.rename(full_current_path, full_new_path)
                current_number += 1
            else:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--toss_id",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    toss_id = args.toss_id
    toss_type = args.type
    if toss_type == 'napkin':
        # napkin_traj_dir = f'/home/cnets-vision/mengti_ws/BundleSDF/dair_pll/assets/bundlesdf_napkin/{toss_id-1}.pt'
        napkin_traj_dir = f'/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_napkin/{toss_id-1}.pt'
        napkin_urdf_dir = "drake/../../../../../../../../../../home/cnets-vision/mengti_ws/BundleSDF/assets/gt_napkin.urdf"
        simulate_toss_with_traj(napkin_traj_dir, napkin_urdf_dir)
        # napkin_dir = f'/home/cnets-vision/mengti_ws/BundleSDF/dair_pll/assets/bundlesdf_napkin'
        # bad_tosses = [5, 9, 10]
        # postprocess(napkin_dir, bad_tosses)
    elif toss_type == 'bottle':
        bottle_dir = f'/home/cnets-vision/mengti_ws/BundleSDF/dair_pll/assets/bundlesdf_bottle'
        bottle_traj_dir = f'/home/cnets-vision/mengti_ws/BundleSDF/dair_pll/assets/bundlesdf_bottle/{toss_id-1}.pt'
        bottle_urdf_dir = "drake/../../../../../../../../../../home/cnets-vision/mengti_ws/BundleSDF/assets/gt_bottle.urdf"
        simulate_toss_with_traj(bottle_traj_dir, bottle_urdf_dir)
    elif toss_type == 'cube':
        cube_traj_dir = f'/home/cnets-vision/mengti_ws/BundleSDF/dair_pll/assets/contactnets_cube/{toss_id-1}.pt'
        simulate_cube_toss_with_traj(None, cube_traj_dir)
    elif toss_type == 'bundlesdf_cube':
        cube_traj_dir = f'/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_cube/{toss_id-1}.pt'
        simulate_cube_toss_with_traj(None, cube_traj_dir)
    elif toss_type == 'milk':
        milk_dir = f'/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_milk'
        milk_traj_dir = f'/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_milk/{toss_id-1}.pt'
        milk_urdf_dir = "drake/../../../../../../../../../../home/cnets-vision/mengti_ws/BundleSDF/assets/gt_bottle.urdf"
        simulate_toss_with_traj(milk_traj_dir, milk_urdf_dir)
    elif toss_type == 'prism':
        prism_dir = f'/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_prism'
        prism_traj_dir = f'/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_prism/{toss_id-1}.pt'
        prism_urdf_dir = "drake/../../../../../../../../../../home/cnets-vision/mengti_ws/BundleSDF/assets/gt_prism.urdf"
        simulate_toss_with_traj(prism_traj_dir, prism_urdf_dir)
    elif toss_type == 'toblerone':
        toblerone_dir = f'/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_toblerone'
        toblerone_traj_dir = f'/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_toblerone/{toss_id-1}.pt'
        toblerone_urdf_dir = "drake/../../../../../../../../../../home/cnets-vision/mengti_ws/BundleSDF/assets/gt_toblerone.urdf"
        simulate_toss_with_traj(toblerone_traj_dir, toblerone_urdf_dir)
    elif toss_type == 'half':
        half_dir = f'/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_half'
        half_traj_dir = f'/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_half/{toss_id-1}.pt'
        half_urdf_dir = "drake/../../../../../../../../../../home/cnets-vision/mengti_ws/BundleSDF/assets/gt_half.urdf"
        simulate_toss_with_traj(half_traj_dir, half_urdf_dir)
        pass
    elif toss_type == 'egg':
        # egg_dir = f'/home/cnets-vision/mengti_ws/BundleSDF/dair_pll/assets/bundlesdf_egg'
        # egg_traj_dir = f'/home/cnets-vision/mengti_ws/BundleSDF/dair_pll/assets/bundlesdf_egg/{toss_id-1}.pt'
        # egg_dir = f'/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_egg'
        # egg_traj_dir = f'/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_egg/{toss_id-1}.pt'
        # simulate_toss_with_traj(egg_traj_dir, egg_urdf_dir)
        pass
    else:
        print('Invalid toss type')