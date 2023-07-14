# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Particle Filter with RM
Author: Hongrui Zheng
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from particle_filter import utils as Utils
from particle_filter.jax_rm import get_dt, get_scan

"""
These flags indicate several variants of the sensor model. Only one of them is used at a time.
"""
VAR_NO_EVAL_SENSOR_MODEL = 0
VAR_CALC_RANGE_MANY_EVAL_SENSOR = 1
VAR_REPEAT_ANGLES_EVAL_SENSOR = 2
VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT = 3
VAR_RADIAL_CDDT_OPTIMIZATIONS = 4

@dataclass
class pf_config:
    angle_step: int = 18
    max_particles: int = 4000
    squash_factor: float = 2.2
    theta_discretization: int = 112
    max_range: float = 10
    fine_timing: float = 0
    z_short: float = 0.01
    z_max: float = 0.07
    z_rand: float = 0.12
    z_hit: float = 0.75
    sigma_hit: float = 8.0
    motion_dispersion_x: float = 0.05
    motion_dispersion_y: float = 0.025
    motion_dispersion_theta: float = 0.25

class ParticleFiler(object):
    """
    This class implements Monte Carlo Localization based on odometry and a laser scanner.
    """

    def __init__(self, config=pf_config()):
        # parameters
        self.config = config

        # various data containers used in the MCL algorithm
        self.max_range_px = None
        self.odometry_data = np.array([0.0, 0.0, 0.0])
        self.laser = None
        self.iters = 0
        self.map_info = None
        self.map_initialized = False
        self.lidar_initialized = False
        self.odom_initialized = False
        self.last_pose = None
        self.laser_angles = None
        self.downsampled_angles = None
        self.range_method = None
        self.last_time = None
        self.last_stamp = None
        self.first_sensor_update = True

        # cache this to avoid memory allocation in motion model
        self.local_deltas = np.zeros((self.config.max_particles, 3))

        # cache this for the sensor model computation
        self.queries = None
        self.ranges = None
        self.tiled_angles = None
        self.sensor_model_table = None

        # particle poses and weights
        self.inferred_pose = None
        self.particle_indices = np.arange(self.config.max_particles)
        self.particles = np.zeros((self.config.max_particles, 3))
        self.weights = np.ones(self.config.max_particles) / float(self.config.max_particles)

        self.precompute_sensor_model()
        self.initialize_global()

        # keep track of speed from input odom
        self.current_speed = 0.0


    def set_omap(self, bitmap, metadata):
        """
        Sets the occupancy map for the pf object
        """
        self.map = bitmap
        # TODO: fix image orientation, see laser_models
        self.dt = get_dt(bitmap, metadata['resolution'])
        self.max_range_px = self.config.max_range / metadata['resolution']


    def lidarCB(self, msg):
        """
        Initializes reused buffers, and stores the relevant laser scanner data for later use.
        """
        if not isinstance(self.laser_angles, np.ndarray):
            self.get_logger().info("...Received first LiDAR message")
            self.laser_angles = np.linspace(
                msg.angle_min, msg.angle_max, len(msg.ranges)
            )
            self.downsampled_angles = np.copy(
                self.laser_angles[0 :: self.ANGLE_STEP]
            ).astype(np.float32)
            self.viz_queries = np.zeros(
                (self.downsampled_angles.shape[0], 3), dtype=np.float32
            )
            self.viz_ranges = np.zeros(
                self.downsampled_angles.shape[0], dtype=np.float32
            )
            self.get_logger().info(str(self.downsampled_angles.shape[0]))

        # store the necessary scanner information for later processing
        self.downsampled_ranges = np.array(msg.ranges[:: self.ANGLE_STEP])
        self.lidar_initialized = True
        # self.update()

    def odomCB(self, msg):
        """
        Store deltas between consecutive odometry messages in the coordinate space of the car.

        Odometry data is accumulated via dead reckoning, so it is very inaccurate on its own.
        """
        position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

        orientation = Utils.quaternion_to_angle(msg.pose.pose.orientation)
        pose = np.array([position[0], position[1], orientation])
        self.current_speed = msg.twist.twist.linear.x

        if isinstance(self.last_pose, np.ndarray):
            # changes in x,y,theta in local coordinate system of the car
            rot = Utils.rotation_matrix(-self.last_pose[2])
            delta = np.array([position - self.last_pose[0:2]]).transpose()
            local_delta = (rot * delta).transpose()

            self.odometry_data = np.array(
                [local_delta[0, 0], local_delta[0, 1], orientation - self.last_pose[2]]
            )
            self.last_pose = pose
            self.last_stamp = msg.header.stamp
            self.odom_initialized = True
        else:
            self.get_logger().info("...Received first Odometry message")
            self.last_pose = pose

        # this topic is slower than lidar, so update every time we receive a message
        self.update()


    def initialize_particles_pose(self, pose):
        """
        Initialize particles in the general region of the provided pose.
        """

        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)
        self.particles[:, 0] = pose.position.x + np.random.normal(
            loc=0.0, scale=0.5, size=self.MAX_PARTICLES
        )
        self.particles[:, 1] = pose.position.y + np.random.normal(
            loc=0.0, scale=0.5, size=self.MAX_PARTICLES
        )
        self.particles[:, 2] = Utils.quaternion_to_angle(
            pose.orientation
        ) + np.random.normal(loc=0.0, scale=0.4, size=self.MAX_PARTICLES)

    def initialize_global(self):
        """
        Spread the particle distribution over the permissible region of the state space.
        """

        # randomize over grid coordinate space
        permissible_x, permissible_y = np.where(self.permissible_region == 1)
        indices = np.random.randint(0, len(permissible_x), size=self.MAX_PARTICLES)

        permissible_states = np.zeros((self.MAX_PARTICLES, 3))
        permissible_states[:, 0] = permissible_y[indices]
        permissible_states[:, 1] = permissible_x[indices]
        permissible_states[:, 2] = np.random.random(self.MAX_PARTICLES) * np.pi * 2.0

        Utils.map_to_world(permissible_states, self.map_info)
        self.particles = permissible_states
        self.weights[:] = 1.0 / self.MAX_PARTICLES

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model. For each discrete computed
        range value, this provides the probability of measuring any (discrete) range.

        This table is indexed by the sensor model at runtime by discretizing the measurements
        and computed ranges from RangeLibc.
        """
        self.get_logger().info("Precomputing sensor model")
        # sensor model constants
        z_short = self.config.z_short
        z_max = self.config.z_max
        z_rand = self.config.z_rand
        z_hit = self.config.z_hit
        sigma_hit = self.config.sigma_hit

        table_width = int(self.max_range_px) + 1
        self.sensor_model_table = np.zeros((table_width, table_width))

        # d is the computed range from RangeLibc
        for d in range(table_width):
            norm = 0.0
            sum_unkown = 0.0
            # r is the observed range from the lidar unit
            for r in range(table_width):
                prob = 0.0
                z = float(r - d)
                # reflects from the intended object
                prob += (
                    z_hit
                    * np.exp(-(z * z) / (2.0 * sigma_hit * sigma_hit))
                    / (sigma_hit * np.sqrt(2.0 * np.pi))
                )

                # observed range is less than the predicted range - short reading
                if r < d:
                    prob += 2.0 * z_short * (d - r) / float(d)

                # erroneous max range measurement
                if int(r) == int(self.max_range_px):
                    prob += z_max

                # random measurement
                if r < int(self.max_range_px):
                    prob += z_rand * 1.0 / float(self.max_range_px)

                norm += prob
                self.sensor_model_table[int(r), int(d)] = prob

            # normalize
            self.sensor_model_table[:, int(d)] /= norm

        # upload the sensor model to RangeLib for ultra fast resolution
        if self.RANGELIB_VAR > 0:
            self.range_method.set_sensor_model(self.sensor_model_table)

    def motion_model(self, proposal_dist, action):
        """
        The motion model applies the odometry to the particle distribution. Since there the odometry
        data is inaccurate, the motion model mixes in gaussian noise to spread out the distribution.

        Vectorized motion model. Computing the motion model over all particles is thousands of times
        faster than doing it for each particle individually due to vectorization and reduction in
        function call overhead

        TODO this could be better, but it works for now
            - fixed random noise is not very realistic
            - ackermann model provides bad estimates at high speed
        """
        # rotate the action into the coordinate space of each particle
        # t1 = time.time()
        cosines = np.cos(proposal_dist[:, 2])
        sines = np.sin(proposal_dist[:, 2])

        self.local_deltas[:, 0] = cosines * action[0] - sines * action[1]
        self.local_deltas[:, 1] = sines * action[0] + cosines * action[1]
        self.local_deltas[:, 2] = action[2]

        proposal_dist[:, :] += self.local_deltas
        proposal_dist[:, 0] += np.random.normal(
            loc=0.0, scale=self.MOTION_DISPERSION_X, size=self.MAX_PARTICLES
        )
        proposal_dist[:, 1] += np.random.normal(
            loc=0.0, scale=self.MOTION_DISPERSION_Y, size=self.MAX_PARTICLES
        )
        proposal_dist[:, 2] += np.random.normal(
            loc=0.0, scale=self.MOTION_DISPERSION_THETA, size=self.MAX_PARTICLES
        )

    def sensor_model(self, proposal_dist, obs, weights):
        """
        This function computes a probablistic weight for each particle in the proposal distribution.
        These weights represent how probable each proposed (x,y,theta) pose is given the measured
        ranges from the lidar scanner.

        There are 4 different variants using various features of RangeLibc for demonstration purposes.
        - VAR_REPEAT_ANGLES_EVAL_SENSOR is the most stable, and is very fast.
        - VAR_NO_EVAL_SENSOR_MODEL directly indexes the precomputed sensor model. This is slow
                                   but it demonstrates what self.range_method.eval_sensor_model does
        - VAR_RADIAL_CDDT_OPTIMIZATIONS is only compatible with CDDT or PCDDT, it implments the radial
                                        optimizations to CDDT which simultaneously performs ray casting
                                        in two directions, reducing the amount of work by roughly a third
        """

        num_rays = self.downsampled_angles.shape[0]
        # only allocate buffers once to avoid slowness
        if self.first_sensor_update:
            if self.RANGELIB_VAR <= 1:
                self.queries = np.zeros(
                    (num_rays * self.MAX_PARTICLES, 3), dtype=np.float32
                )
            else:
                self.queries = np.zeros((self.MAX_PARTICLES, 3), dtype=np.float32)

            self.ranges = np.zeros(num_rays * self.MAX_PARTICLES, dtype=np.float32)
            self.tiled_angles = np.tile(self.downsampled_angles, self.MAX_PARTICLES)
            self.first_sensor_update = False

        if self.RANGELIB_VAR == VAR_RADIAL_CDDT_OPTIMIZATIONS:
            if "cddt" in self.WHICH_RM:
                self.queries[:, :] = proposal_dist[:, :]
                self.range_method.calc_range_many_radial_optimized(
                    num_rays,
                    self.downsampled_angles[0],
                    self.downsampled_angles[-1],
                    self.queries,
                    self.ranges,
                )

                # evaluate the sensor model
                self.range_method.eval_sensor_model(
                    obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES
                )
                # apply the squash factor
                self.weights = np.power(self.weights, self.INV_SQUASH_FACTOR)
            else:
                self.get_logger().info(
                    "Cannot use radial optimizations with non-CDDT based methods, use rangelib_variant 2"
                )
        elif self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT:
            self.queries[:, :] = proposal_dist[:, :]
            self.range_method.calc_range_repeat_angles_eval_sensor_model(
                self.queries, self.downsampled_angles, obs, self.weights
            )
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
        elif self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR:
            if self.SHOW_FINE_TIMING:
                t_start = time.time()
            # this version demonstrates what this would look like with coordinate space conversion pushed to rangelib
            self.queries[:, :] = proposal_dist[:, :]
            if self.SHOW_FINE_TIMING:
                t_init = time.time()
            self.range_method.calc_range_repeat_angles(
                self.queries, self.downsampled_angles, self.ranges
            )
            if self.SHOW_FINE_TIMING:
                t_range = time.time()
            # evaluate the sensor model on the GPU
            self.range_method.eval_sensor_model(
                obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES
            )
            if self.SHOW_FINE_TIMING:
                t_eval = time.time()
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
            if self.SHOW_FINE_TIMING:
                t_squash = time.time()
                t_total = (t_squash - t_start) / 100.0

            if self.SHOW_FINE_TIMING and self.iters % 10 == 0:
                self.get_logger().info(
                    str(
                        [
                            "sensor_model: init: ",
                            np.round((t_init - t_start) / t_total, 2),
                            "range:",
                            np.round((t_range - t_init) / t_total, 2),
                            "eval:",
                            np.round((t_eval - t_range) / t_total, 2),
                            "squash:",
                            np.round((t_squash - t_eval) / t_total, 2),
                        ]
                    )
                )
        elif self.RANGELIB_VAR == VAR_CALC_RANGE_MANY_EVAL_SENSOR:
            # this version demonstrates what this would look like with coordinate space conversion pushed to rangelib
            # this part is inefficient since it requires a lot of effort to construct this redundant array
            self.queries[:, 0] = np.repeat(proposal_dist[:, 0], num_rays)
            self.queries[:, 1] = np.repeat(proposal_dist[:, 1], num_rays)
            self.queries[:, 2] = np.repeat(proposal_dist[:, 2], num_rays)
            self.queries[:, 2] += self.tiled_angles

            self.range_method.calc_range_many(self.queries, self.ranges)

            # evaluate the sensor model on the GPU
            self.range_method.eval_sensor_model(
                obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES
            )
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
        elif self.RANGELIB_VAR == VAR_NO_EVAL_SENSOR_MODEL:
            # this version directly uses the sensor model in Python, at a significant computational cost
            self.queries[:, 0] = np.repeat(proposal_dist[:, 0], num_rays)
            self.queries[:, 1] = np.repeat(proposal_dist[:, 1], num_rays)
            self.queries[:, 2] = np.repeat(proposal_dist[:, 2], num_rays)
            self.queries[:, 2] += self.tiled_angles

            # compute the ranges for all the particles in a single functon call
            self.range_method.calc_range_many(self.queries, self.ranges)

            # resolve the sensor model by discretizing and indexing into the precomputed table
            obs /= float(self.map_info.resolution)
            ranges = self.ranges / float(self.map_info.resolution)
            obs[obs > self.max_range_px] = self.max_range_px
            ranges[ranges > self.max_range_px] = self.max_range_px

            intobs = np.rint(obs).astype(np.uint16)
            intrng = np.rint(ranges).astype(np.uint16)

            # compute the weight for each particle
            for i in range(self.MAX_PARTICLES):
                weight = np.product(
                    self.sensor_model_table[
                        intobs, intrng[i * num_rays : (i + 1) * num_rays]
                    ]
                )
                weight = np.power(weight, self.INV_SQUASH_FACTOR)
                weights[i] = weight
        else:
            self.get_logger().info("PLEASE SET rangelib_variant PARAM to 0-4")

    def MCL(self, a, o):
        """
        Performs one step of Monte Carlo Localization.
            1. resample particle distribution to form the proposal distribution
            2. apply the motion model
            3. apply the sensor model
            4. normalize particle weights

        This is in the critical path of code execution, so it is optimized for speed.
        """
        if self.SHOW_FINE_TIMING:
            t = time.time()
        # draw the proposal distribution from the old particles
        proposal_indices = np.random.choice(
            self.particle_indices, self.MAX_PARTICLES, p=self.weights
        )
        proposal_distribution = self.particles[proposal_indices, :]
        if self.SHOW_FINE_TIMING:
            t_propose = time.time()

        # compute the motion model to update the proposal distribution
        self.motion_model(proposal_distribution, a)
        if self.SHOW_FINE_TIMING:
            t_motion = time.time()

        # compute the sensor model
        self.sensor_model(proposal_distribution, o, self.weights)
        if self.SHOW_FINE_TIMING:
            t_sensor = time.time()

        # normalize importance weights
        self.weights /= np.sum(self.weights)
        if self.SHOW_FINE_TIMING:
            t_norm = time.time()
            t_total = (t_norm - t) / 100.0

        if self.SHOW_FINE_TIMING and self.iters % 10 == 0:
            self.get_logger().info(
                str(
                    [
                        "MCL: propose: ",
                        np.round((t_propose - t) / t_total, 2),
                        "motion:",
                        np.round((t_motion - t_propose) / t_total, 2),
                        "sensor:",
                        np.round((t_sensor - t_motion) / t_total, 2),
                        "norm:",
                        np.round((t_norm - t_sensor) / t_total, 2),
                    ]
                )
            )

        # save the particles
        self.particles = proposal_distribution

    def expected_pose(self):
        # returns the expected value of the pose given the particle distribution
        return np.dot(self.particles.transpose(), self.weights)

    def update(self):
        """
        Apply the MCL function to update particle filter state.

        Ensures the state is correctly initialized, and acquires the state lock before proceeding.
        """
        if self.lidar_initialized and self.odom_initialized and self.map_initialized:
            if self.state_lock.locked():
                self.get_logger().info("Concurrency error avoided")
            else:
                self.state_lock.acquire()
                self.timer.tick()
                self.iters += 1

                t1 = time.time()
                observation = np.copy(self.downsampled_ranges).astype(np.float32)
                action = np.copy(self.odometry_data)
                self.odometry_data = np.zeros(3)

                # run the MCL update algorithm
                self.MCL(action, observation)

                # compute the expected value of the robot pose
                self.inferred_pose = self.expected_pose()
                self.state_lock.release()
                t2 = time.time()

                # publish transformation frame based on inferred pose
                self.publish_tf(self.inferred_pose, self.last_stamp)

                # this is for tracking particle filter speed
                ips = 1.0 / (t2 - t1)
                self.smoothing.append(ips)
                if self.iters % 10 == 0:
                    self.get_logger().info(
                        str(
                            [
                                "iters per sec:",
                                int(self.timer.fps()),
                                " possible:",
                                int(self.smoothing.mean()),
                            ]
                        )
                    )

                self.visualize()


# import argparse
# import sys
# parser = argparse.ArgumentParser(description='Particle filter.')
# parser.add_argument('--config', help='Path to yaml file containing config parameters. Helpful for calling node directly with Python for profiling.')

# def load_params_from_yaml(fp):
#     from yaml import load
#     with open(fp, 'r') as infile:
#         yaml_data = load(infile)
#         for param in yaml_data:
#             print 'param:', param, ':', yaml_data[param]
#             rospy.set_param('~'+param, yaml_data[param])

# # this function can be used to generate flame graphs easily
# def make_flamegraph(filterx=None):
#     import flamegraph, os
#     perf_log_path = os.path.join(os.path.dirname(__file__), '../tmp/perf.log')
#     flamegraph.start_profile_thread(fd=open(perf_log_path, 'w'),
#                                     filter=filterx,
#                                     interval=0.001)


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFiler()
    rclpy.spin(pf)


if __name__ == "__main__":
    main()

# if __name__=='__main__':
#     rospy.init_node('particle_filter')

#     args,_ = parser.parse_known_args()
#     if args.config:
#         load_params_from_yaml(args.config)

#     # make_flamegraph(r'update')

#     pf = ParticleFiler()
#     rospy.spin()
