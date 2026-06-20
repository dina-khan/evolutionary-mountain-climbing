import os
import genome
import sys
import creature
import pybullet as p
import time
import numpy as np

def make_arena(arena_size=20, wall_height=1):
    wall_thickness = 0.5

    floor_collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX, 
        halfExtents=[arena_size/2, arena_size/2, wall_thickness]
    )

    floor_visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX, 
        halfExtents=[arena_size/2, arena_size/2, wall_thickness], 
        rgbaColor=[1, 1, 0, 1]
    )

    floor_body = p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=floor_collision_shape, 
        baseVisualShapeIndex=floor_visual_shape, 
        basePosition=[0, 0, -wall_thickness]
    )

    wall_collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX, 
        halfExtents=[arena_size/2, wall_thickness/2, wall_height/2]
    )

    wall_visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX, 
        halfExtents=[arena_size/2, wall_thickness/2, wall_height/2], 
        rgbaColor=[0.7, 0.7, 0.7, 1]
    )  

    p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=wall_collision_shape, 
        baseVisualShapeIndex=wall_visual_shape, 
        basePosition=[0, arena_size/2, wall_height/2]
    )

    p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=wall_collision_shape, 
        baseVisualShapeIndex=wall_visual_shape, 
        basePosition=[0, -arena_size/2, wall_height/2]
    )

    wall_collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX, 
        halfExtents=[wall_thickness/2, arena_size/2, wall_height/2]
    )

    wall_visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX, 
        halfExtents=[wall_thickness/2, arena_size/2, wall_height/2], 
        rgbaColor=[0.7, 0.7, 0.7, 1]
    )  

    p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=wall_collision_shape, 
        baseVisualShapeIndex=wall_visual_shape, 
        basePosition=[arena_size/2, 0, wall_height/2]
    )

    p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=wall_collision_shape, 
        baseVisualShapeIndex=wall_visual_shape, 
        basePosition=[-arena_size/2, 0, wall_height/2]
    )

def main(csv_file):
    assert os.path.exists(csv_file), f"Tried to load {csv_file} but it does not exist"

    p.connect(p.GUI)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # resetting camera target and rotation to focus on elite creature
    p.resetDebugVisualizerCamera(
        cameraDistance=10,
        cameraYaw=40,
        cameraPitch=-45,
        cameraTargetPosition=[-3.0, -7.0, 3.0]
    )

    p.setGravity(0, 0, -10)

    make_arena(arena_size=20)

    mountain_position = (0, 0, -1)  
    mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))

    p.setAdditionalSearchPath('shapes/')
    p.loadURDF("gaussian_pyramid.urdf", 
               mountain_position, 
               mountain_orientation, 
               useFixedBase=1)

    dna = genome.Genome.from_csv(csv_file)
    cr = creature.Creature(gene_count=len(dna))
    cr.update_dna(dna)

    with open("test.urdf", "w") as f:
        f.write(cr.to_xml())

    rob1 = p.loadURDF("test.urdf")

    p.resetBasePositionAndOrientation(rob1, 
                                      [-3.0, -7.0, 1.0], 
                                      [0, 0, 0, 1])
    
    start_pos, orn = p.getBasePositionAndOrientation(rob1)

    # motor output signals will be scaled by this 
    # value to give velocity in radians
    motor_velocity_scale = 30.0

    # motors exert force to reach the given velocity
    # the force will be limited to this value 
    motor_force = 400.0

    # motors will be updated after 6 steps
    motor_update_steps = 6 

    # 1 simulation step (wait_time) is 1/240
    # motors will be updated after 6 steps
    # this gives the time in seconds for 6 simulation steps
    motor_update_time = motor_update_steps / 240.0

    elapsed_time = 0.0

    # 1 simulation step (wait_time) is 1/240
    wait_time = 1 / 240.0

    # The simulation will run for 40 seconds
    total_time = 40.0
    step = 0

    while True:
        p.stepSimulation()
        step += 1

        # motors will be updated every 6 steps
        if step % motor_update_steps == 0:
            # getting all creature motors
            motors = cr.get_motors()
            assert len(motors) == p.getNumJoints(rob1), "Something went wrong"

            # Iterating over all creature joints
            for jid in range(p.getNumJoints(rob1)):
                mode = p.VELOCITY_CONTROL

                # scaling motor output signal to get velocity
                vel = motors[jid].get_output(motor_update_time) * motor_velocity_scale

                # updating motor
                p.setJointMotorControl2(rob1,
                                        jid,
                                        controlMode=mode,
                                        targetVelocity=vel,
                                        force=motor_force)

            # getting current creature position
            pos, orn = p.getBasePositionAndOrientation(rob1)

            # printing updated information about creature as it moves
            # distance covered, radius from center mountain, and height
            print(f"distance: {float(np.linalg.norm(np.asarray(start_pos) - np.asarray(pos)))}, radius: {float(np.sqrt(pos[0] * pos[0] + pos[1] * pos[1]))}, height: {float(pos[2])}")

        # stepping through wait time
        time.sleep(wait_time)

        # breaking from loop if simulation time is complete
        elapsed_time += wait_time
        if elapsed_time > total_time:
            break

    os.remove("test.urdf")

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python realtime_from_csv_arena.py result_logs\<experiment folder>\<run folder>\<generation elite csv>.csv"
    main(sys.argv[1])
