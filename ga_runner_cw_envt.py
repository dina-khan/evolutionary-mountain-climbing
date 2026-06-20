import os
import csv
import math
import numpy as np
import pybullet as p
import genome
import creature
import population

class Simulation:
    def __init__(self, sim_id=0):
        self.sim_id = sim_id
        self.physicsClientId = p.connect(p.DIRECT)

        self.floor_id = None
        self.mountain_id = None

    def _make_arena(self, pid, arena_size=20, wall_height=1):
        wall_thickness = 0.5

        floor_collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[arena_size / 2, arena_size / 2, wall_thickness],
            physicsClientId=pid
        )

        floor_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[arena_size / 2, arena_size / 2, wall_thickness],
            rgbaColor=[1, 1, 0, 1],
            physicsClientId=pid
        )

        # assigning the floor body to self.floor_id
        self.floor_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_collision_shape,
            baseVisualShapeIndex=floor_visual_shape,
            basePosition=[0, 0, -wall_thickness],
            physicsClientId=pid
        )

        wall_collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2],
            physicsClientId=pid
        )
        wall_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2],
            rgbaColor=[0.7, 0.7, 0.7, 1],
            physicsClientId=pid
        )
        p.createMultiBody(
            0,
            wall_collision_shape, wall_visual_shape,
            [0,  arena_size / 2, wall_height / 2], 
            physicsClientId=pid
        )
        p.createMultiBody(
            0, 
            wall_collision_shape, 
            wall_visual_shape,
            [0, -arena_size / 2, wall_height / 2], 
            physicsClientId=pid
        )

        wall_collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2],
            physicsClientId=pid
        )
        wall_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2],
            rgbaColor=[0.7, 0.7, 0.7, 1],
            physicsClientId=pid
        )
        p.createMultiBody(
            0, 
            wall_collision_shape, 
            wall_visual_shape,
            [arena_size / 2, 0, wall_height / 2], 
            physicsClientId=pid
        )
        p.createMultiBody(
            0, 
            wall_collision_shape, 
            wall_visual_shape,
            [-arena_size / 2, 0, wall_height / 2], 
            physicsClientId=pid
        )

    # This function returns the height of the platform underneath the creature,
    # given the creature body and its horizontal position
    def _platform_underneath_height(self, pid, x, y, creature):

        # setting starting height and ending height of ray detection
        start_z, end_z = 60, -60

        # Iterating so detection may be continued if the ray hits creature
        for i in range(15):
            # ray detection at creature horizontal position from height start_z to end_z
            detection = p.rayTest([x, y, start_z], [x, y, end_z], physicsClientId=pid)[0]

            # if the body detected is the creature,
            # moving the ray start below the creature and continuing detection
            if detection[0] == creature:
                # setting starting height of ray below creature
                start_z = detection[3][2] - 0.02

                # returning if the ray start height goes too low
                if start_z <= end_z:
                    return None
                
                # continuing detection below creature
                continue

            # returning height if the body detected is the floor or the mountain
            if detection[0] == self.floor_id or detection[0] == self.mountain_id:
                return detection[3][2]
            
            return None

    # This function returns true if the creature 
    # is in contact with the mountain
    def _check_on_mountain(self, pid, creature):

        # this pybullet function returns all the points
        # where the creature collides with the mountain body
        creature_mountain_collisions = p.getContactPoints(
            bodyA=creature, # creature body
            bodyB=self.mountain_id, # mountain body
            physicsClientId=pid
        )

        # returning true if there is at least one contact point
        if len(creature_mountain_collisions) > 0:
            return True
        
        return False

    # this function updates all motors of the creature
    def _update_motors(
            self, 
            pid, # physics client
            creature_body, # pybullet creature
            cr, # creature object
            motor_update_time, # time difference to ensure correct motor waveform phase
            motor_velocity_scale, # to scale motor output signal into velocity
            motor_force # limiting force allowed to reach target velocity
    ):
        # storing motors and number of motors
        motors = cr.get_motors()
        motors_number = p.getNumJoints(creature_body, physicsClientId=pid)

        # represents the creature's current distributed motors effort
        motors_effort = 0.0

        # iterating over all motors
        for jid in range(motors_number):

            # Getting current motor signal output using motor class, 
            # passing motor update time to ensure correct waveform phase.
            # Scaling motor output signal to get velocity
            motor_target_velocity = motors[jid].get_output(motor_update_time) * motor_velocity_scale

            # Adding this motor's effort to the total 
            motors_effort += abs(motor_target_velocity)

            # updating pybullet motor
            p.setJointMotorControl2(
                creature_body, 
                jid, 
                controlMode=p.VELOCITY_CONTROL, 
                targetVelocity=motor_target_velocity,
                #limiting force allowed to reach target velocity
                force=motor_force, 
                physicsClientId=pid
            )

        # returning combined effort of motors distributed over the number of motors
        return motors_effort / max(1, motors_number)
           
    # This function runs the given creature and 
    # returns its fitness and other metrics for plotting
    def get_fitness_run_creature(self, cr, iterations=2400):
        # storing simulation physics client id
        pid = self.physicsClientId

        # resetting simulation
        p.resetSimulation(physicsClientId=pid)

        # setting physics engine
        p.setPhysicsEngineParameter(
            enableFileCaching=0, 
            fixedTimeStep=1.0 / 240.0, 
            physicsClientId=pid
        )

        # setting gravity
        p.setGravity(0, 0, -10, physicsClientId=pid)

        # making all motors phase consistent
        for motor in cr.get_motors():
            motor.phase = 0.0


        # making the arena with the physics client id
        self._make_arena(pid)

        # setting mountain position and orientation
        mountain_position = (0, 0, -1)  
        mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))

        # path to search for mountain urdf
        p.setAdditionalSearchPath('shapes/', physicsClientId=pid)

        # storing the mountain body in self.mountain_id
        self.mountain_id = p.loadURDF(
            "gaussian_pyramid.urdf",
            basePosition=mountain_position,
            baseOrientation=mountain_orientation,
            useFixedBase=1,
            physicsClientId=pid
        )

        # storing creature URDF xml in temporary file
        xml_file = f"temp{self.sim_id}.urdf"
        with open(xml_file, "w") as f:
            f.write(cr.to_xml())

        # loading URDF creature body into pybullet
        creature_body = p.loadURDF(xml_file, physicsClientId=pid)

        # setting a fixed starting position and orientation for the creature
        p.resetBasePositionAndOrientation(
            creature_body, 
            [-3.0, -7.0, 1.0], 
            [0, 0, 0, 1], 
            physicsClientId=pid
        )


        ## STORING PROPERTIES USED TO CALCULATE FITNESS REWARDS AND PENALTIES ##

        # initial x y z position of the creature
        (initial_x, initial_y, initial_z), orn = p.getBasePositionAndOrientation(creature_body, physicsClientId=pid)
        # initial radius of the creature from the mountain center
        initial_radius = math.sqrt(initial_x * initial_x + initial_y * initial_y)
        # height of surface initially below creature
        initial_platform_height = self._platform_underneath_height(pid, initial_x, initial_y, creature_body) or 0.0

        # storing last updated x y z values
        # useful for checking differences
        last_x, last_y, last_z = initial_x, initial_y, initial_z

        # storing last updated radius of the creature from the mountain center
        # useful e.g. to check if the creature is moving towards the mountain
        last_radius = initial_radius

        # storing the last updated height of the surface below the creature
        # useful e.g to check if the creature is going uphill
        last_platform_height = initial_platform_height

        # keeping track of the highest platform height the creature reached
        highest_platform_height = initial_platform_height

        # keeping track of the the closest horizontal distance 
        # the creature reached from the mountain center
        min_radius = initial_radius

        # total horizontal distance covered
        total_horizontal_movement = 0.0

        # total effort by motors during simulation
        total_motors_effort = 0.0

        # effort by motors during current simulation step
        current_motors_effort = 0.0

        # limit for maximum distance creature motors can exert
        motor_force = 400.0

        # creature motor output signals will be 
        # scaled by this value for targer velocity
        motor_velocity_scale = 30.0

        # motors will be updated after 6 steps
        motor_update_steps = 6 

        # 1 simulation step (wait_time) is 1/240
        # motors will be updated after 6 steps
        # this gives the time in seconds for 6 simulation steps
        motor_update_time = motor_update_steps / 240.0
        
        # height creature is allowed to reach before penalizing for flying
        flying_threshold = 2.0

        # The creature motors may be putting extra 'useless' effort to shake the creature in place 
        # without noticeably moving the creature towards a direction.
        # Threshold of useless motors effort allowed without penalty.
        useless_motors_effort_threshold = 5.0

        decreasing_radius_reward = 0.0 # rewarding any reduction in horizontal distance from mountain center
        closest_radius_reward = 0.0 # rewarding closest horizontal radius reached towards mountain center
        lifting_reward = 0.0 # rewarding any increase in platform height reached
        highest_lift_reward = 0.0 # rewarding highest platform reached
        mountain_collisions_reward = 0.0 # rewarding all mountain collisions
        mountain_climbing_reward = 0.0 # rewarding ascent while colliding with mountain

        increasing_radius_penalty = 0.0 # penilizing any increase in horizontal distance from mountain center
        running_wall_penalty = 0.0 # penilizing any wall proximity during simulation
        ending_wall_penalty = 0.0 # heavily penalizing wall proximity when simulation ends
        long_path_penalty = 0.0 # penalizing longer paths taken by creature to cover straight distance
        useless_motors_effort_penalty = 0.0 # penalizing effort by motors that causes useless shaking
        flying_penalty = 0.0 # penalizing flying
        links_penalty = 0.0 # penalizing excessive links growth

        ## STORING PROPERTIES USED TO CALCULATE METRICS FOR GRAPHS AND TABLES ##
        flying_steps = 0                   
        mountain_collision_steps = 0        
        wall_proximity_steps = 0               
        useless_motors_effort_total = 0.0


        # iterating over simulation steps
        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)

            # motors will be updated every 6 steps
            if step % motor_update_steps == 0:

                # updating motors and storing the current 
                # combined effort distributed over all motors
                current_motors_effort = self._update_motors(
                    pid=pid,
                    creature_body=creature_body,
                    cr=cr,
                    motor_update_time=motor_update_time,
                    motor_velocity_scale=motor_velocity_scale,
                    motor_force=motor_force
                )



                # adding the motors effort from last motors update to total motors effort
                total_motors_effort += current_motors_effort

            ## CALCULATING REWARDS AND PENALTIES DURING EACH SIMULATION STEP ##

            # storing new creature position 
            (current_x, current_y, current_z), orn = p.getBasePositionAndOrientation(creature_body, physicsClientId=pid)
   
            # horizontal movement since last time creature position was updated
            change_horizontal_movement = math.sqrt((current_x - last_x) ** 2 + (current_y - last_y) ** 2)

            # total horizontal distance covered
            total_horizontal_movement += change_horizontal_movement

            # updating horizontal distance of creature from center of mountain
            current_radius = math.sqrt(current_x * current_x + current_y * current_y)

            # storing distance from the closest wall
            wall_distance = min(10.0 - abs(current_x), 10.0 - abs(current_y))
            # imposing penalty if too close to wall
            if wall_distance < 2.0:
                wall_proximity_steps += 1

                # Adding further penalty to wall proximity penalty,
                # while the creature remains near the wall.
                # Penalty which increases as wall distance decreases
                running_wall_penalty += (2.0 - wall_distance) / 2.0
            
            # If the creature is currently colliding with the mountain,
            # incrementing the number of simulation steps in which the 
            # creature collided with the mountain.
            # This will be used to reward each step of colliding with the mountain
            if self._check_on_mountain(pid, creature_body):
                mountain_collisions_reward += 1
                mountain_collision_steps += 1

            # Setting minimum radius to current radius if the creature
            # moved horizontally closer to the mountain than ever before.
            # This will be used for rewarding minimum radius reached
            if current_radius < min_radius:
                min_radius = current_radius

            # storing the difference in creature's horizontal radius from mountain center
            change_radius = current_radius - last_radius

            # if the radius is increased, 
            # a penalty is imposed for going away from the mountain
            if change_radius > 0.0:
                # the penalty is scaled with the difference in radius 
                # the penalty is greater if more simulation time has passed
                increasing_radius_penalty += change_radius * (2.0 + 1.5 * step / float(max(1, iterations)))
            # if the radius is decreases,
            # a reward is given for going towards the mountain
            else:
                # the penalty is scaled with the difference in radius 
                # the penalty is greater if less simulation time has passed
                decreasing_radius_reward += abs(change_radius) * (2.0 - 0.5 * step / float(max(1, iterations)))

            # Storing height of platform currently beneath creature
            current_platform_height = self._platform_underneath_height(pid, current_x, current_y, creature_body)
            if current_platform_height is None:
                # Updating the creature's last position, radius to current values 
                last_x, last_y, last_z = current_x, current_y, current_z
                last_radius= current_radius
                continue
            
            # incrementing the flying penalty if the creature is flying, 
            # if its current height from the platform beneath is greater than the flying threshold
            if current_z - current_platform_height > flying_threshold:
                flying_penalty += 1

                flying_steps += 1

            # if the current platform beneath the creature is 
            # higher than the highest recorded platform reached, 
            if current_platform_height > highest_platform_height:
                # updating the highest platform height reached by the creature
                highest_platform_height = current_platform_height

            # Storing the difference between the heights of the 
            # surfaces currently and previously beneath the creature
            # The lift height is positive if the creature reaches a higher mountain surface
            lift_height = current_platform_height - last_platform_height

            # If the creature has lifted itself in this simulation step
            if lift_height > 0.0:
                # Ensuring creature is not flying, 
                # if the difference between the creature height 
                # and the surface below the threshold
                if current_z - current_platform_height < flying_threshold:
                    # Adding additional surface height to the lift reward
                    lifting_reward += lift_height

                    # further reward if the creature is lifting itself while 
                    # colliding with the mountain
                    if self._check_on_mountain(pid, creature_body):
                        mountain_climbing_reward += lift_height * 2.0

            # if the creature is almost stationary
            if (change_horizontal_movement < 0.01) and (abs(lift_height) <= 0.0001) and (current_z - current_platform_height < flying_threshold):
                # checking if the creature is uselessly wobbling, 
                # if current motors effort is more than the allowed
                # almost useless motors effort
                if current_motors_effort > useless_motors_effort_threshold:
                    # adding the extra effort over the threshold to the useless motors effort penalty
                    useless_motors_effort_penalty += (current_motors_effort - useless_motors_effort_threshold)

                    extra = (current_motors_effort - useless_motors_effort_threshold)
                    useless_motors_effort_total += extra

            # Updating the height of the last platform beneath 
            # the creature to the height of the current platform
            # Updating the creature's last position, radius to current values 
            last_x, last_y, last_z = current_x, current_y, current_z
            last_radius= current_radius
            last_platform_height = current_platform_height

        ## UPDATING REWARDS AND PENALTIES AFTER SIMULATION STEPS ARE COMPLETE ##

        # storing the final x y z position of the creature after the simulation is complete
        (ending_x, ending_y, ending_z), orn = p.getBasePositionAndOrientation(creature_body, physicsClientId=pid)
 
        # storing direct horizontal distance between the creature's start position and final position
        straight_horizontal_displacement = math.sqrt((ending_x - initial_x) ** 2 + (ending_y - initial_y) ** 2)

        # if the creature's displacement is greater than 1
        if straight_horizontal_displacement > 1.0:
            # ratio of the total horizontal distance covered by creature over its displacement
            # this increases as the creature takes longer paths
            long_path_ratio = total_horizontal_movement / straight_horizontal_displacement

            # if the ratio of longer paths taken is greater than 2
            if long_path_ratio > 2.0:
                # scaled penalty for the creature taking longer paths than the ratio
                long_path_penalty = (long_path_ratio - 2.0) * 1500

        # if total effort by motors in simulation is greater than 1,
        if total_motors_effort > 1.0:
            # scaling penalty with ratio of almost useless motors effort over total motors effort
            useless_motors_effort_penalty = 2000.0 * useless_motors_effort_penalty / total_motors_effort

        # scaled total of all penalties for remaining near wall during the running simulation
        running_wall_penalty = 4000.0 * (running_wall_penalty / float(max(1, iterations)))

        # storing final distance from closest wall after all simulation steps are complete
        final_wall_distance = min(10.0 - abs(ending_x), 10.0 - abs(ending_y))

        # if the creature is too close to the wall at the end of the simulation
        if final_wall_distance < 2.0:
            # scaled penalty that is greater if the final wall distance is smaller
            ending_wall_penalty = 6000.0 * (2.0 - final_wall_distance) / 2.0

        # if the creature reached a mountain surface higher than initial surface height
        if highest_platform_height > initial_platform_height:
            # reward scaled with difference between the highest level reached and initial surface height
            highest_lift_reward = 5000.0 * (highest_platform_height - initial_platform_height)

        # if the creature decreased its horizontal radius from the mountain at some point 
        if min_radius < initial_radius:
            # heavy reward scaled with difference between initial radius and minimum radius
            closest_radius_reward = 10000 * (initial_radius - min_radius)

        # reward for every step which decreased horizontal radius from mountain
        decreasing_radius_reward = 5000.0 * decreasing_radius_reward

        # heavy penalty for every step which decreased horizontal radius from mountain
        increasing_radius_penalty = 10000.0 * increasing_radius_penalty

        # A penalty is imposed for each link generated by the creature
        # This prevents the creature from cheating by growing too many limbs
        links_penalty = 1000.0 * float(len(cr.get_expanded_links()))

        # scaled total penalty for each simulation step during which the creature 
        # is flying
        flying_penalty = 300.0 * (flying_penalty / float(max(1, iterations)))
        
        # scaled total reward for each simulation step during which the creature 
        # is colliding with the mountain
        mountain_collisions_reward = 4000.0 * (mountain_collisions_reward / float(max(1, iterations)))

        # scaled total reward for each simulation step during which the creature 
        # is lifting itself while colliding with the mountain
        mountain_climbing_reward = 15000.0 * (mountain_climbing_reward / float(max(1, iterations)))

        # scaled total reward for each simulation step during which the creature is lifting itself
        lifting_reward = 1500.0 * (lifting_reward / float(max(1, iterations)))
        
        ## FITNESS AS SUM OF CALCULATED REWARDS/PENALTIES ##

        fitness = (
            # rewarding closest horizontal radius reached towards mountain center
            closest_radius_reward 
            # rewarding any reduction in horizontal radius from mountain center
            + decreasing_radius_reward 
            # rewarding all mountain collisions
            + mountain_collisions_reward 
            # rewarding highest platform reached
            + highest_lift_reward
            # rewarding any increase in platform height reached
            + lifting_reward
            # rewarding ascent while colliding with mountain
            + mountain_climbing_reward
            # penilizing any increase in horizontal radius from mountain center
            - increasing_radius_penalty
            # penalizing longer paths taken by creature to cover straight distance
            - long_path_penalty
            # penalizing effort by motors that causes useless shaking
            - useless_motors_effort_penalty
            # penalizing flying
            - flying_penalty
            # penilizing any wall proximity during simulation
            - running_wall_penalty
            # heavily penalizing wall proximity when simulation ends
            - ending_wall_penalty
            # penalizing excessive links growth
            - links_penalty
        )

        # calculating metrics for graph and table plotting
        if straight_horizontal_displacement > 1.0:
            long_path_ratio_raw = float(total_horizontal_movement / straight_horizontal_displacement)
        else:
            long_path_ratio_raw = 0.0

        flying_ratio = float(flying_steps / float(max(1, iterations)))
        mountain_collision_ratio = float(mountain_collision_steps / float(max(1, iterations)))
        wall_proximity_ratio = float(wall_proximity_steps / float(max(1, iterations)))

        # metrics dictionary
        metrics = {
            "fitness": float(max(0.0, fitness)),
            "min_radius": float(min_radius),
            "long_path_ratio_raw": float(long_path_ratio_raw),
            "highest_platform_height": float(highest_platform_height),
            "mountain_collision_ratio": float(mountain_collision_ratio),
            "flying_ratio": float(flying_ratio),
            "wall_proximity_ratio": float(wall_proximity_ratio),
            "useless_motors_effort_total": float(useless_motors_effort_total),
            "total_motor_effort": float(total_motors_effort),
            "links": float(len(cr.get_expanded_links())),
        }
        # removing the creature urdf
        os.remove(xml_file)

        # returning fitness and metrics 
        return float(max(0.0, fitness)), metrics

# This function runs the genetic algorithm
def run_genetic_algorithm_experiment(
    generations_count=100,
    pop_size=50,
    gene_count=3,
    experiment_name="experiment",
    run=1
):
    # Creating a folder to log results of all experiment runs
    os.makedirs("result_logs", exist_ok=True)

    # Creating the numbered experiment folder within the results folder
    experiment_folder = os.path.join("result_logs", experiment_name, f"run{run}")
    os.makedirs(experiment_folder, exist_ok=True)
    
    # Creating a csv file within current experiment folder to 
    # store fitness statistics with generations
    metrics_path = os.path.join(experiment_folder, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "generation",

            # elite metrics
            "elite_fitness",
            "mean_fitness",
            "elite_min_radius",
            "elite_highest_platform_reached",
            "elite_mountain_collision_ratio",
            "elite_flying_ratio",
            "elite_wall_proximity_ratio",
            "elite_long_path_ratio",
            "elite_total_motor_effort",
            "elite_useless_motor_effort",
            "elite_links",

            # mean creature metrics
            "mean_min_radius",
            "mean_highest_platform_reached",
            "mean_mountain_contact_ratio",
            "mean_flying_ratio",
            "mean_wall_proximity_ratio",
            "mean_long_path_ratio",
            "mean_total_motor_effort",
            "mean_useless_motor_effort",
            "mean_links",
        ])


    # Creating population and simulation
    pop = population.Population(pop_size=pop_size, gene_count=gene_count)
    sim = Simulation()

    # this keeps track of whether the genetic algorithm is in 
    # evolve only motor genes mode or evolve all genes mode
    evolve_only_motors_mode = False   

    # Calling this function of the genome class returns the indices of genes to be mutated
    # It takes the gene specification from the creature class as input, 
    # along with the evolve mode (only motors/all genes)
    mutation_gene_indices = genome.Genome.get_gene_indices_to_be_mutated(
        creature.Creature(gene_count=gene_count).spec, 
        evolve_only_motors = evolve_only_motors_mode
    )


    # Iterating over each generation
    for generation in range(generations_count):
        fits = []
        links = []
        metrics_list = []

        # Iterating over each creature in the population
        for cr in pop.creatures:
            # The creature fitness is returned by simulation function get_fitness_run_creature
            # Storing creature fitness and metrics
            fit, m = sim.get_fitness_run_creature(cr)
            fits.append(fit)
            metrics_list.append(m)

            # Storing expanded links from creature class
            links.append(len(cr.get_expanded_links()))

        fits = np.array(fits, dtype=float)
        links = np.array(links, dtype=float)

        print("gen", generation, "best fitness", int(np.max(fits)))

        # tuning shrink/grow mutation rates
        shrink_mutate_rate = 0.20
        grow_mutate_rate = 0.15

        # Making point mutation rate and amount range scale inversely 
        # with generation so mutation decreases in later generations 
        point_mutate_rate = 0.40 / (1 + 0.03 * generation)
        point_mutate_range = 0.50 / (1 + 0.02 * generation)

        # Storing the creature from population with maximum fitness
        elite = pop.creatures[int(np.argmax(fits))]
        elite_metric_values = metrics_list[int(np.argmax(fits))]

        # population mean creature metrics
        mean_min_radius = float(np.mean([m["min_radius"] for m in metrics_list]))
        mean_highest_platform = float(np.mean([m["highest_platform_height"] for m in metrics_list]))
        mean_mountain_contact = float(np.mean([m["mountain_collision_ratio"] for m in metrics_list]))
        mean_flying = float(np.mean([m["flying_ratio"] for m in metrics_list]))
        mean_wall_proximity_ratio = float(np.mean([m["wall_proximity_ratio"] for m in metrics_list]))
        mean_long_path_ratio = float(np.mean([m["long_path_ratio_raw"] for m in metrics_list]))
        mean_useless_effort = float(np.mean([m["useless_motors_effort_total"] for m in metrics_list]))
        mean_total_motor_effort = float(np.mean([m["total_motor_effort"] for m in metrics_list]))
        mean_links = float(np.mean([m["links"] for m in metrics_list]))

        # Writing fitness statistics of current generation 
        # as a new row in the csv file
        with open(metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                generation,
                
                # elite creature metrics
                float(np.max(fits)),
                float(np.mean(fits)),
                elite_metric_values["min_radius"],
                elite_metric_values["highest_platform_height"],
                elite_metric_values["mountain_collision_ratio"],
                elite_metric_values["flying_ratio"],
                elite_metric_values["wall_proximity_ratio"],
                elite_metric_values["long_path_ratio_raw"],
                elite_metric_values["total_motor_effort"],
                elite_metric_values["useless_motors_effort_total"],
                elite_metric_values["links"],

                # mean creature metrics
                mean_min_radius,
                mean_highest_platform,
                mean_mountain_contact,
                mean_flying,
                mean_wall_proximity_ratio,
                mean_long_path_ratio,
                mean_total_motor_effort,
                mean_useless_effort,
                mean_links,
            ])

        new_creatures = []

        # Copying the elite creature's dna into a new creature
        # which will be added as it is to the next generation
        new_elite = creature.Creature(gene_count=gene_count)
        new_elite.update_dna(np.array(elite.dna, copy=True))
        new_creatures.append(new_elite)

        # Saving the elite creature's dna to a csv file,
        genome.Genome.to_csv(
            elite.dna,
            # Storing the file within the current experiment run folder,
            # and numbering it by the generation
            os.path.join(experiment_folder, f"generation{generation}_best.csv")
        )

        # Storing fitness by passing fitnesses to population class
        fit_map = population.Population.get_fitness_map(fits)

        # Loop which creates creatures of the next generation
        while len(new_creatures) < len(pop.creatures):
            # Selecting parent indices using the fitmap
            p1_ind = population.Population.select_parent(fit_map)
            p2_ind = population.Population.select_parent(fit_map)

            # Storing the parent creatures
            p1 = pop.creatures[p1_ind]
            p2 = pop.creatures[p2_ind]

            # Crossing over parents dna using genome class
            dna = genome.Genome.crossover(p1.dna, p2.dna)

            # Mutating dna
            dna = genome.Genome.point_mutate(
                dna, 
                rate=point_mutate_rate,
                amount=point_mutate_range, 

                # Passing the indices of genes to be mutated to the point 
                # mutate function. If the genetic algorithm is in evolve 
                # only motors mode, only the indices of motor related genes 
                # are passed. This allows the genetic algorithm to mutate 
                # either all genes or motor related genes.
                mutation_gene_indices=mutation_gene_indices)

            # Only applying grow/shrink mutate if the genetic algorithm
            # is not in evolve only motors mode, it is is in evolve all genes mode
            if evolve_only_motors_mode == False:

                dna = genome.Genome.shrink_mutate(dna, rate=shrink_mutate_rate)
                dna = genome.Genome.grow_mutate(dna, rate=grow_mutate_rate)

            # Creating a new creature with the resulting dna
            new_creature = creature.Creature(gene_count=gene_count)
            new_creature.update_dna(dna)
            new_creatures.append(new_creature)

        # Setting the next generation's population to 
        # the new creatures determined from previous generation
        pop.creatures = new_creatures

    print(f"Experiment results logged in {experiment_folder}")
    return experiment_folder

# Running the genetic algorithm 3 times for each experiment setting
if __name__ == "__main__":
    for run in [1, 2, 3]:
        run_genetic_algorithm_experiment(
            generations_count=100,
            pop_size=30,
            gene_count=3,
            experiment_name="base",
            run=run
        )
