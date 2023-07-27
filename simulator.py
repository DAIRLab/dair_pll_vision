import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import yaml
import sys

import pydrake
from pydrake.all import StartMeshcat, RandomGenerator, UniformlyRandomRotationMatrix, BodyIndex
from pydrake.common import FindResourceOrThrow
from pydrake.common.eigen_geometry import Quaternion, AngleAxis, Isometry3
from pydrake.geometry import (
    Box,
    HalfSpace,
    SceneGraph,
    Sphere,
)
from pydrake.math import (RollPitchYaw, RigidTransform)
from pydrake.multibody.tree import (
    PrismaticJoint,
    SpatialInertia,
    UniformGravityFieldElement,
    UnitInertia
)
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    MultibodyPlant
)

from pydrake.forwarddiff import gradient
from pydrake.multibody.parsing import Parser
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers.ipopt import (IpoptSolver)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import MeshcatVisualizer, MeshcatVisualizerParams

urdf_file = "assets/contactnets_cube.urdf"
def AddShape(plant, shape, name, mass=1, mu=1, color=[0.5, 0.5, 0.9, 1.0]):
    instance = plant.AddModelInstance(name)
    if isinstance(shape, Box):
        inertia = UnitInertia.SolidBox(
            shape.width(), shape.depth(), shape.height()
        )
    else:
        print("Only support Box instance for now.")
        return
    body = plant.AddRigidBody(
        name,
        instance,
        SpatialInertia(
            mass=mass, p_PScm_E=np.array([0.0, 0.0, 0.0]), G_SP_E=inertia
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

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    rng = np.random.default_rng(135)  # this is for python
    generator = RandomGenerator(rng.integers(0, 1000))
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=0.0001))
    shape = Box(0.1, 0.1, 0.1)
    name = 'cube'
    AddShape(plant, shape, name, mass=1, mu=1, color=[0.5, 0.5, 0.9, 1.0])
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
    plant.SetPositions(plant_context, q0)
    z = 0.2
    for body_index in plant.GetFloatingBaseBodies():
        tf = RigidTransform(
            UniformlyRandomRotationMatrix(generator),
            [rng.uniform(0.35, 0.65), rng.uniform(-0.12, 0.28), z],
        )
        plant.SetFreeBodyPose(plant_context, plant.get_body(body_index), tf)
        z += 0.1
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)
    q0_final = plant.GetPositions(plant_context).copy()
    print(q0, q0_final)
    while True:
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        cube_position = plant.EvalBodyPoseInWorld(plant_context, plant.get_body(BodyIndex(0))).translation()
        print(cube_position)
        if cube_position[2] <= 0.01:  
            # The cube has landed, so reset the simulation
            simulator.get_mutable_context().SetTime(0.)
            plant.SetPositions(plant_context, q0)
            for body_index in plant.GetFloatingBaseBodies():
                tf = RigidTransform(
                    UniformlyRandomRotationMatrix(generator),
                    [rng.uniform(0.35, 0.65), rng.uniform(-0.12, 0.28), z],
                )
                plant.SetFreeBodyPose(plant_context, plant.get_body(body_index), tf)
            simulator.Initialize()