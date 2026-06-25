# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from copy import deepcopy
from os import path

import threading
import time

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid
from curobo.geom.types import Cylinder
from curobo.geom.types import Mesh
from curobo.geom.types import Sphere
from curobo.geom.types import VoxelGrid as CuVoxelGrid
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState as CuJointState
from curobo.util.logger import setup_curobo_logger
from curobo.wrap.reacher.motion_gen import MotionGen
from curobo.wrap.reacher.motion_gen import MotionGenConfig
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from curobo.wrap.reacher.motion_gen import MotionGenStatus
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from isaac_ros_cumotion.update_kinematics import get_robot_config
from isaac_ros_cumotion.update_kinematics import UpdateLinkSpheresServer
from isaac_ros_cumotion_python_utils.utils import \
    get_grid_center, get_grid_min_corner, get_grid_size, is_grid_valid, \
    load_grid_corners_from_workspace_file
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import CollisionObject
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.msg import RobotTrajectory
import numpy as np
from nvblox_msgs.srv import EsdfAndGradients
import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
import torch
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker


def _mem_report(node, tag):
    """Intra-process GPU memory breakdown (gated by CUMOTION_MEM_PROFILE env).

    Decomposes the single per-PID number nvtop/NVML reports into:
      live_tensors  = torch.cuda.memory_allocated  (tensors actually in use)
      torch_pool    = torch.cuda.memory_reserved    (torch caching allocator)
      slack         = torch_pool - live_tensors     (reserved-but-unused)
      ctx+libs+graphs = NVML_total - torch_pool      (CUDA context, cuBLAS/cuDNN
                        cubins, Warp kernels, non-torch CUDA-graph pools) — the
                        part torch's own counters (and thus any external tool) miss.
    No-op unless CUMOTION_MEM_PROFILE is set, so it never affects normal runs.
    """
    import os
    if not os.environ.get('CUMOTION_MEM_PROFILE'):
        return
    import subprocess
    try:
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 2**20
        resv = torch.cuda.memory_reserved() / 2**20
        total = float('nan')
        pid = os.getpid()
        out = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5).stdout
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
                total = float(parts[1])
                break
        node.get_logger().info(
            f'[MEMPROF {tag}] live_tensors={alloc:.0f}MB '
            f'torch_pool={resv:.0f}MB(slack={resv - alloc:.0f}) '
            f'nvsmi_total={total:.0f}MB ctx+libs+graphs={total - resv:.0f}MB')
    except Exception as e:  # profiling must never break the node
        node.get_logger().warn(f'[MEMPROF {tag}] failed: {e}')


def _record_mem_history():
    """Start torch's allocation recorder (gated by CUMOTION_MEM_SNAPSHOT).

    Every subsequent cudaMalloc is tagged with its Python call stack, so a later
    _dump_snapshot lets us attribute live VRAM to the *function* that allocated
    it — the 'which function uses the most GPU memory' question. Call as early as
    possible so no allocation is missed."""
    import os
    if not os.environ.get('CUMOTION_MEM_SNAPSHOT'):
        return
    try:
        torch.cuda.memory._record_memory_history(max_entries=200000)
    except TypeError:  # older torch signature
        try:
            torch.cuda.memory._record_memory_history(True)
        except Exception:
            pass
    except Exception:
        pass


def _dump_mem_snapshot(node, path):
    """Pickle the allocation snapshot to `path` (gated by CUMOTION_MEM_SNAPSHOT).
    Analyze offline with scripts/analyze_cumotion_snapshot.py (no GPU needed)."""
    import os
    if not os.environ.get('CUMOTION_MEM_SNAPSHOT'):
        return
    try:
        torch.cuda.memory._dump_snapshot(path)
        node.get_logger().info(f'[MEMSNAP] dumped {path}')
    except Exception as e:
        node.get_logger().warn(f'[MEMSNAP] dump failed: {e}')


class CumotionActionServer(Node):

    def __init__(self):
        super().__init__('cumotion_action_server')
        _mem_report(self, '00_init_start')
        _record_mem_history()
        self.tensor_args = TensorDeviceType()
        self.declare_parameter('robot', 'ur5e.yml')
        self.declare_parameter('urdf_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('yml_file_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('time_dilation_factor', 0.5)
        self.declare_parameter('max_attempts', 10)
        self.declare_parameter('num_graph_seeds', 6)
        # graph_file: optional override for curobo's graph-search config (cspace PRM +
        # steering). Empty string => use curobo's default graph.yml (upstream behaviour).
        # Non-empty => absolute path resolved by launch; passed straight into
        # MotionGenConfig.load_from_robot_config(graph_file=...). Used to tune
        # base_dt / steer_delta_buffer when graph search returns through-collision paths.
        self.declare_parameter('graph_file', '')
        self.declare_parameter('num_trajopt_seeds', 6)
        self.declare_parameter('include_trajopt_retract_seed', True)
        self.declare_parameter('num_trajopt_time_steps', 32)
        self.declare_parameter('trajopt_finetune_iters', 400)
        self.declare_parameter('interpolation_dt', 0.025)
        # Interpolated-trajectory buffer length (steps). curobo's default is
        # 5000; collision / self-collision / FK metric buffers are allocated
        # at batch x interpolation_steps x n_spheres, which makes this the
        # planner's largest VRAM knob (measured 6.3 GB -> 2.5 GB going
        # 5000 -> 1024 with 275 spheres, zero latency change). Required
        # length = num_trajopt_time_steps * maximum_trajectory_dt /
        # interpolation_dt (384 on the production config; worst case ~480
        # with finetune dt relaxation). On overflow curobo grows the buffer
        # and re-captures CUDA graphs (one-time latency blip, not a failure).
        self.declare_parameter('interpolation_steps', 5000)
        self.declare_parameter('maximum_trajectory_dt', 0.15)
        self.declare_parameter('collision_cache_mesh', 20)
        self.declare_parameter('collision_cache_cuboid', 20)
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('read_esdf_world', False)
        # plan_on_empty_esdf: when read_esdf_world is on and the ESDF service
        # returns an all-unobserved grid (nothing within nvblox integration
        # range -> the workspace is genuinely clear), plan against an
        # obstacle-free voxel world instead of aborting. Default False keeps
        # upstream behavior (abort + retry, appropriate during sensor warmup);
        # set True for a robot that must stay drivable in open space.
        self.declare_parameter('plan_on_empty_esdf', False)
        self.declare_parameter('publish_curobo_world_as_voxels', False)
        self.declare_parameter('add_ground_plane', False)
        self.declare_parameter('publish_voxel_size', 0.05)
        self.declare_parameter('max_publish_voxels', 500000)
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('tool_frame', rclpy.Parameter.Type.STRING)

        # The grid_center_m and grid_size_m parameters are loaded from the workspace file
        # if the workspace_file_path is set and valid.
        self.declare_parameter('workspace_file_path', '')
        self.declare_parameter('grid_center_m', [0.0, 0.0, 0.0])
        self.declare_parameter('grid_size_m', [2.0, 2.0, 2.0])
        self.declare_parameter('update_esdf_on_request', True)
        self.declare_parameter('use_aabb_on_request', True)

        self.declare_parameter('esdf_service_name', '/nvblox_node/get_esdf_and_gradient')
        self.declare_parameter('enable_curobo_debug_mode', False)
        self.declare_parameter('override_moveit_scaling_factors', False)
        self.declare_parameter('update_link_sphere_server',
                               'planner_attach_object')
        self.declare_parameter('publish_iteration_trajectories', False)
        self.declare_parameter('iteration_publish_stride', 1)
        self.declare_parameter('iteration_animation_period_s', 1.0)
        self.declare_parameter('trajectory_viz_topic', '/planned_trajectory_viz')
        self.declare_parameter('trajectory_viz_frame', '')
        debug_mode = (
            self.get_parameter('enable_curobo_debug_mode').get_parameter_value().bool_value
        )
        if debug_mode:
            setup_curobo_logger('info')
        else:
            setup_curobo_logger('warning')

        self.__voxel_pub = self.create_publisher(Marker, '/curobo/voxels', 10)

        self._publish_iter_trajs = (
            self.get_parameter('publish_iteration_trajectories')
            .get_parameter_value().bool_value
        )
        stride = (
            self.get_parameter('iteration_publish_stride')
            .get_parameter_value().integer_value
        )
        self._iter_publish_stride = max(1, stride)
        self._iter_animation_period = max(
            0.0,
            self.get_parameter('iteration_animation_period_s')
            .get_parameter_value().double_value,
        )
        traj_viz_topic = (
            self.get_parameter('trajectory_viz_topic')
            .get_parameter_value().string_value
        )
        self._traj_viz_frame_override = (
            self.get_parameter('trajectory_viz_frame')
            .get_parameter_value().string_value
        )
        # Generous queue depth: a single plan publishes (per-iter overlays + final)
        # all back-to-back, and rviz with reliable QoS will drop messages if the
        # publisher buffer overflows.
        self._traj_viz_pub = self.create_publisher(Marker, traj_viz_topic, 200)
        self._traj_viz_iter_count = 0
        self._traj_viz_anim_thread = None
        self._traj_viz_anim_stop = threading.Event()
        self._traj_viz_lock = threading.Lock()
        self.planner_busy = False
        self.lock = threading.Lock()

        self.__robot_file = self.get_parameter('robot').get_parameter_value().string_value

        try:
            self.__urdf_path = self.get_parameter('urdf_path')
            self.__urdf_path = self.__urdf_path.get_parameter_value().string_value
            if self.__urdf_path == '':
                self.__urdf_path = None
        except rclpy.exceptions.ParameterUninitializedException:
            self.__urdf_path = None

        try:
            self.__yml_path = self.get_parameter('yml_file_path')
            self.__yml_path = self.__yml_path.get_parameter_value().string_value
            if self.__yml_path == '':
                self.__yml_path = None
        except rclpy.exceptions.ParameterUninitializedException:
            self.__yml_path = None

        # If a YAML path is provided, override other XRDF/YAML file name
        if self.__yml_path is not None:
            self.__robot_file = self.__yml_path
        try:
            self.__tool_frame = self.get_parameter('tool_frame')
            self.__tool_frame = self.__tool_frame.get_parameter_value().string_value
            if self.__tool_frame == '':
                self.__tool_frame = None
        except rclpy.exceptions.ParameterUninitializedException:
            self.__tool_frame = None

        self.__joint_states_topic = (
            self.get_parameter('joint_states_topic').get_parameter_value().string_value
        )
        self.__add_ground_plane = (
            self.get_parameter('add_ground_plane').get_parameter_value().bool_value
        )
        self.__override_moveit_scaling_factors = (
            self.get_parameter('override_moveit_scaling_factors').get_parameter_value().bool_value
        )

        # Motion generation parameters

        self.__max_attempts = (
            self.get_parameter('max_attempts').get_parameter_value().integer_value
        )
        self.__num_graph_seeds = (
            self.get_parameter('num_graph_seeds').get_parameter_value().integer_value
        )
        self.__graph_file = (
            self.get_parameter('graph_file').get_parameter_value().string_value
        )
        self.__num_trajopt_seeds = (
            self.get_parameter('num_trajopt_seeds').get_parameter_value().integer_value
        )
        self.__num_trajopt_time_steps = (
            self.get_parameter('num_trajopt_time_steps').get_parameter_value().integer_value
        )
        self.__trajopt_finetune_iters = (
            self.get_parameter('trajopt_finetune_iters').get_parameter_value().integer_value
        )
        self.__interpolation_dt = (
            self.get_parameter('interpolation_dt').get_parameter_value().double_value
        )
        self.__interpolation_steps = (
            self.get_parameter('interpolation_steps')
            .get_parameter_value().integer_value
        )
        self.__maximum_trajectory_dt = (
            self.get_parameter('maximum_trajectory_dt').get_parameter_value().double_value
        )

        include_trajopt_retract_seed = (
            self.get_parameter('include_trajopt_retract_seed').get_parameter_value().bool_value
        )
        if include_trajopt_retract_seed:
            self.__num_trajopt_noisy_seeds = 1
            self.__trajopt_seed_ratio = {'linear': 1.0}
        else:
            self.__num_trajopt_noisy_seeds = 2
            self.__trajopt_seed_ratio = {'linear': 0.5, 'bias': 0.5}

        collision_cache_cuboid = (
            self.get_parameter('collision_cache_cuboid').get_parameter_value().integer_value
        )
        collision_cache_mesh = (
            self.get_parameter('collision_cache_mesh').get_parameter_value().integer_value
        )
        self.__collision_cache = {
            'obb': collision_cache_cuboid,
            'mesh': collision_cache_mesh
        }

        # ESDF service

        self.__read_esdf_grid = (
            self.get_parameter('read_esdf_world').get_parameter_value().bool_value
        )
        self.__plan_on_empty_esdf = (
            self.get_parameter('plan_on_empty_esdf').get_parameter_value().bool_value
        )
        self.__publish_curobo_world_as_voxels = (
            self.get_parameter('publish_curobo_world_as_voxels').get_parameter_value().bool_value
        )
        self.__grid_center_m = (
            self.get_parameter('grid_center_m').get_parameter_value().double_array_value
        )
        self.__max_publish_voxels = (
            self.get_parameter('max_publish_voxels').get_parameter_value().integer_value
        )
        self.__workspace_file_path = (
            self.get_parameter('workspace_file_path').get_parameter_value().string_value
        )
        self.__grid_size_m = (
            self.get_parameter('grid_size_m').get_parameter_value().double_array_value
        )
        self.__update_esdf_on_request = (
            self.get_parameter('update_esdf_on_request').get_parameter_value().bool_value
        )
        self.__use_aabb_on_request = (
            self.get_parameter('use_aabb_on_request').get_parameter_value().bool_value
        )
        self.__publish_voxel_size = (
            self.get_parameter('publish_voxel_size').get_parameter_value().double_value
        )
        self.__voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        self._update_link_sphere_server = (
            self.get_parameter(
                'update_link_sphere_server').get_parameter_value().string_value
        )
        self.__esdf_client = None
        self.__esdf_req = None

        # Setup the grid position and dimension.
        if path.exists(self.__workspace_file_path):
            self.get_logger().info(
                f'Loading grid center and dims from workspace file: {self.__workspace_file_path}.')
            min_corner, max_corner = load_grid_corners_from_workspace_file(
                self.__workspace_file_path)
            self.__grid_size_m = get_grid_size(min_corner, max_corner, self.__voxel_size)
            self.__grid_center_m = get_grid_center(min_corner, self.__grid_size_m)

            self.get_logger().info(
                f'Loaded grid dims: {self.__grid_size_m}, ' + f'voxel size: {self.__voxel_size}')
        else:
            self.get_logger().info(
                'Loading grid position and dims from grid_center_m and grid_size_m parameters.')

        if is_grid_valid(self.__grid_size_m, self.__voxel_size):
            self.get_logger().fatal('Number of voxels should be at least 1 in every dimension.')
            raise SystemExit

        if self.__read_esdf_grid:
            esdf_service_name = (
                self.get_parameter('esdf_service_name').get_parameter_value().string_value
            )

            esdf_service_cb_group = MutuallyExclusiveCallbackGroup()
            self.__esdf_client = self.create_client(
                EsdfAndGradients, esdf_service_name, callback_group=esdf_service_cb_group
            )
            while not self.__esdf_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(
                    f'Service({esdf_service_name}) not available, waiting again...'
                )
            self.__esdf_req = EsdfAndGradients.Request()

        self.load_motion_gen()
        _mem_report(self, '01_after_load_motion_gen')
        self.warmup()
        _mem_report(self, '02_after_warmup')
        _dump_mem_snapshot(self, '/tmp/cumotion_mem_after_warmup.pickle')
        self.__mem_snap_after_plan_done = False
        # Periodic sampler captures the steady state + the step-up the first real
        # plan adds (CUDA-graph capture + trajopt/MPPI peak the allocator pins).
        self.create_timer(5.0, lambda: _mem_report(self, 'periodic'))
        self.__query_count = 0
        self.__tensor_args = self.motion_gen.tensor_args
        self.subscription = self.create_subscription(
            JointState, self.__joint_states_topic, self.js_callback, 10
        )
        self.__js_buffer = None

        # Call on_timer every 0.01 seconds
        self.timer = self.create_timer(0.01, self.on_timer)

        self.__update_link_spheres_server = UpdateLinkSpheresServer(
            server_node=self,
            action_name=self._update_link_sphere_server,
            robot_kinematics=self.motion_gen.kinematics,
            robot_base_frame=self.__robot_base_frame
        )
        self._action_server = ActionServer(
            self, MoveGroup, 'cumotion/move_group', self.execute_callback
        )

    def _try_acquire_planner(self):
        """Atomically reserve the (non-reentrant) MotionGen instance.

        Returns True and marks the planner busy if it was free, otherwise
        returns False without changing state. The check-and-set is performed
        under ``self.lock`` so two concurrent action callbacks (the executor is
        MultiThreaded) can never both observe ``planner_busy is False`` and
        proceed into ``motion_gen.plan_*`` simultaneously. cuRobo MotionGen is
        not reentrant; overlapping plan calls corrupt state or crash.
        """
        with self.lock:
            if self.planner_busy:
                return False
            self.planner_busy = True
            return True

    def _release_planner(self):
        """Release the MotionGen reservation taken by _try_acquire_planner."""
        with self.lock:
            self.planner_busy = False

    def js_callback(self, msg):
        self.__js_buffer = {
            'joint_names': msg.name,
            'position': msg.position,
            'velocity': msg.velocity,
        }

    def load_motion_gen(self):
        tensor_args = self.tensor_args
        world_file = WorldConfig.from_dict(
            {
                'cuboid': {
                    'table': {
                        'pose': [0, 0, -0.05, 1, 0, 0, 0],  # x, y, z, qw, qx, qy, qz
                        'dims': [2.0, 2.0, 0.1],
                    }
                },
                'voxel': {
                    'world_voxel': {
                        'dims': self.__grid_size_m,
                        'pose': [0, 0, 0, 1, 0, 0, 0],  # x, y, z, qw, qx, qy, qz
                        'voxel_size': self.__voxel_size,
                        'feature_dtype': torch.bfloat16,
                    },
                },
            }
        )

        robot_config = get_robot_config(
            robot_file=self.__robot_file,
            urdf_file_path=self.__urdf_path,
            logger=self.get_logger()
        )

        robot_dict = robot_config['robot_cfg']
        graph_file_kwargs = (
            {'graph_file': self.__graph_file} if self.__graph_file else {}
        )
        if self.__graph_file:
            self.get_logger().info(
                f'cuMotion graph search using override config: {self.__graph_file}'
            )
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_dict,
            world_file,
            tensor_args,
            num_graph_seeds=self.__num_graph_seeds,
            num_trajopt_seeds=self.__num_trajopt_seeds,
            num_trajopt_noisy_seeds=self.__num_trajopt_noisy_seeds,
            trajopt_tsteps=self.__num_trajopt_time_steps,
            trajopt_seed_ratio=self.__trajopt_seed_ratio,
            interpolation_dt=self.__interpolation_dt,
            interpolation_steps=self.__interpolation_steps,
            maximum_trajectory_dt=self.__maximum_trajectory_dt,
            collision_cache=self.__collision_cache,
            collision_checker_type=CollisionCheckerType.VOXEL,
            ee_link_name=self.__tool_frame,
            finetune_trajopt_iters=self.__trajopt_finetune_iters,
            store_trajopt_debug=self._publish_iter_trajs,
            use_cuda_graph=not self._publish_iter_trajs,
            **graph_file_kwargs,
        )

        motion_gen = MotionGen(motion_gen_config)
        self.motion_gen = motion_gen
        self.__robot_base_frame = self.motion_gen.kinematics.base_link

        self.__world_collision = self.motion_gen.world_coll_checker
        if not self.__add_ground_plane:
            self.motion_gen.clear_world_cache()
        self.__cumotion_grid_shape = self.__world_collision.get_voxel_grid(
            'world_voxel').get_grid_shape()[0]

    def warmup(self):
        self.get_logger().info('warming up cuMotion, wait until ready')
        self.motion_gen.warmup(enable_graph=True)
        self.get_logger().info('cuMotion is ready for planning queries!')

    def on_timer(self):
        with self.lock:
            if self.__js_buffer is None:
                return

            js = np.copy(self.__js_buffer['position'])
            j_names = deepcopy(self.__js_buffer['joint_names'])

        self.__update_link_spheres_server.publish_all_active_spheres(
            robot_joint_states=js,
            robot_joint_names=j_names,
            tensor_args=self.__tensor_args,
            rgb=[0.0, 1.0, 1.0, 1.0]
        )

    def update_voxel_grid(self):
        self.get_logger().info('Calling ESDF service')

        # Get the AABB
        min_corner = get_grid_min_corner(self.__grid_center_m, self.__grid_size_m)
        aabb_min = Point()
        aabb_min.x = min_corner[0]
        aabb_min.y = min_corner[1]
        aabb_min.z = min_corner[2]
        aabb_size = Vector3()
        aabb_size.x = self.__grid_size_m[0]
        aabb_size.y = self.__grid_size_m[1]
        aabb_size.z = self.__grid_size_m[2]

        # Request the esdf grid
        esdf_future = self.send_request(aabb_min, aabb_size)
        while not esdf_future.done():
            time.sleep(0.001)
        response = esdf_future.result()
        if not response.success:
            self.get_logger().info('ESDF request failed, try again after few seconds.')
            return False
        esdf_grid = self.get_esdf_voxel_grid(response)
        if torch.max(esdf_grid.feature_tensor) <= (-1000.0 + 0.5 * self.__voxel_size + 1e-5):
            # Every voxel is the nvblox "unobserved" sentinel. Two causes look
            # identical here: (a) the workspace is genuinely clear (nothing
            # within projective_integrator_max_integration_distance_m of the
            # wrist cam), or (b) the sensor/service is still warming up. The
            # call itself succeeded, so this is not a service failure.
            if not self.__plan_on_empty_esdf:
                self.get_logger().error(
                    'ESDF empty and plan_on_empty_esdf is false -- aborting; '
                    'set plan_on_empty_esdf:=true to plan against a clear world.')
                return False
            # plan_on_empty_esdf: treat empty as a clear world. get_esdf_voxel_grid
            # maps the sentinel to a large free distance, and curobo already
            # treats unobserved voxels as free in every partial grid it builds,
            # so feeding the all-free grid through is consistent (not new unsafe
            # behavior) -- it also clears any stale obstacle from the prior plan.
            # Self-collision and PlanningScene collision objects still apply.
            self.get_logger().warn(
                'ESDF empty (nothing within nvblox integration range); '
                'planning against an obstacle-free voxel world.')
        self.__world_collision.update_voxel_data(esdf_grid)
        self.get_logger().info('Updated ESDF grid')
        return True

    def send_request(self, aabb_min_m, aabb_size_m):
        self.__esdf_req.visualize_esdf = True
        self.__esdf_req.update_esdf = self.__update_esdf_on_request
        self.__esdf_req.use_aabb = self.__use_aabb_on_request
        self.__esdf_req.frame_id = self.__robot_base_frame
        self.__esdf_req.aabb_min_m = aabb_min_m
        self.__esdf_req.aabb_size_m = aabb_size_m
        self.get_logger().info(
            f'ESDF  req = {self.__esdf_req.aabb_min_m}, {self.__esdf_req.aabb_size_m}'
        )
        esdf_future = self.__esdf_client.call_async(self.__esdf_req)

        return esdf_future

    def get_esdf_voxel_grid(self, esdf_data):
        esdf_voxel_size = esdf_data.voxel_size_m
        if abs(esdf_voxel_size - self.__voxel_size) > 1e-4:
            self.get_logger().fatal(
                'Voxel size of esdf array is not equal to requested voxel_size, '
                f'{esdf_voxel_size} vs. {self.__voxel_size}')
            raise SystemExit

        # Get the esdf and gradient data
        esdf_array = esdf_data.esdf_and_gradients
        array_shape = [
            esdf_array.layout.dim[0].size,
            esdf_array.layout.dim[1].size,
            esdf_array.layout.dim[2].size,
        ]
        array_data = np.array(esdf_array.data, dtype=np.float32)
        if (array_data.shape[0] <= 0):
            self.get_logger().fatal(
                'array shape is zero: ' + str(array_data.shape)
            )
            raise SystemExit
        array_data = torch.as_tensor(array_data)

        # Verify the grid shape
        if array_shape != self.__cumotion_grid_shape:
            self.get_logger().fatal(
                'Shape of received esdf voxel grid does not match the cumotion grid shape, '
                f'{array_shape} vs. {self.__cumotion_grid_shape}')
            raise SystemExit

        # Get the origin of the grid
        grid_origin = [
            esdf_data.origin_m.x,
            esdf_data.origin_m.y,
            esdf_data.origin_m.z,
        ]
        # The grid position is defined as the center point of the grid.
        grid_center_m = get_grid_center(grid_origin, self.__grid_size_m)

        # Array data is reshaped to x y z channels
        array_data = array_data.view(array_shape[0], array_shape[1], array_shape[2]).contiguous()

        # Array is squeezed to 1 dimension
        array_data = array_data.reshape(-1, 1)

        # nvblox assigns a value of -1000.0 for unobserved voxels, making it positive
        array_data[array_data < -999.9] = 1000.0

        # nvblox uses negative distance inside obstacles, cuRobo needs the opposite:
        array_data = -1.0 * array_data

        # nvblox treats surface voxels as distance = 0.0, while cuRobo treats
        # distance = 0.0 as not in collision. Adding an offset.
        array_data += 0.5 * self.__voxel_size

        esdf_grid = CuVoxelGrid(
            name='world_voxel',
            dims=self.__grid_size_m,
            pose=grid_center_m + [1, 0.0, 0.0, 0.0],  # x, y, z, qw, qx, qy, qz
            voxel_size=self.__voxel_size,
            feature_dtype=torch.float32,
            feature_tensor=array_data,
        )

        return esdf_grid

    def get_cumotion_collision_object(self, mv_object: CollisionObject):
        objs = []
        pose = mv_object.pose

        world_pose = [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        ]
        world_pose = Pose.from_list(world_pose)
        supported_objects = True
        if len(mv_object.primitives) > 0:
            for k in range(len(mv_object.primitives)):
                pose = mv_object.primitive_poses[k]
                primitive_pose = [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                ]
                object_pose = world_pose.multiply(Pose.from_list(primitive_pose)).tolist()

                if mv_object.primitives[k].type == SolidPrimitive.BOX:
                    # cuboid:
                    dims = mv_object.primitives[k].dimensions
                    obj = Cuboid(
                        name=str(mv_object.id) + '_' + str(k) + '_cuboid',
                        pose=object_pose,
                        dims=dims,
                    )
                    objs.append(obj)
                elif mv_object.primitives[k].type == SolidPrimitive.SPHERE:
                    # sphere:
                    radius = mv_object.primitives[k].dimensions[
                        mv_object.primitives[k].SPHERE_RADIUS
                    ]
                    obj = Sphere(
                        name=str(mv_object.id) + '_' + str(k) + '_sphere',
                        pose=object_pose,
                        radius=radius,
                    )
                    objs.append(obj)
                elif mv_object.primitives[k].type == SolidPrimitive.CYLINDER:
                    # cylinder:
                    cyl_height = mv_object.primitives[k].dimensions[
                        mv_object.primitives[k].CYLINDER_HEIGHT
                    ]
                    cyl_radius = mv_object.primitives[k].dimensions[
                        mv_object.primitives[k].CYLINDER_RADIUS
                    ]
                    obj = Cylinder(
                        name=str(mv_object.id) + '_' + str(k) + '_cylinder',
                        pose=object_pose,
                        height=cyl_height,
                        radius=cyl_radius,
                    )
                    objs.append(obj)
                elif mv_object.primitives[k].type == SolidPrimitive.CONE:
                    self.get_logger().error('Cone primitive is not supported')
                    supported_objects = False
                else:
                    self.get_logger().error('Unknown primitive type')
                    supported_objects = False
        if len(mv_object.meshes) > 0:
            for k in range(len(mv_object.meshes)):
                pose = mv_object.mesh_poses[k]
                mesh_pose = [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                ]
                object_pose = world_pose.multiply(Pose.from_list(mesh_pose)).tolist()
                verts = mv_object.meshes[k].vertices
                verts = [[v.x, v.y, v.z] for v in verts]
                tris = [
                    [v.vertex_indices[0], v.vertex_indices[1], v.vertex_indices[2]]
                    for v in mv_object.meshes[k].triangles
                ]

                obj = Mesh(
                    name=str(mv_object.id) + '_' + str(len(objs)) + '_mesh',
                    pose=object_pose,
                    vertices=verts,
                    faces=tris,
                )
                objs.append(obj)
        return objs, supported_objects

    def get_joint_trajectory(self, js: CuJointState, dt: float):
        traj = RobotTrajectory()
        cmd_traj = JointTrajectory()
        q_traj = js.position.cpu().view(-1, js.position.shape[-1]).numpy()
        vel = js.velocity.cpu().view(-1, js.position.shape[-1]).numpy()
        acc = js.acceleration.view(-1, js.position.shape[-1]).cpu().numpy()
        for i in range(len(q_traj)):
            traj_pt = JointTrajectoryPoint()
            traj_pt.positions = q_traj[i].tolist()
            if js is not None and i < len(vel):
                traj_pt.velocities = vel[i].tolist()
            if js is not None and i < len(acc):
                traj_pt.accelerations = acc[i].tolist()
            time_d = rclpy.time.Duration(seconds=i * dt).to_msg()
            traj_pt.time_from_start = time_d
            cmd_traj.points.append(traj_pt)
        cmd_traj.joint_names = js.joint_names
        cmd_traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_trajectory = cmd_traj
        return traj

    def update_world_objects(self, moveit_objects):
        world_update_status = True
        if len(moveit_objects) > 0:
            cuboid_list = []
            sphere_list = []
            cylinder_list = []
            mesh_list = []
            for i, obj in enumerate(moveit_objects):
                cumotion_objects, obj_update_status = self.get_cumotion_collision_object(obj)
                # Accumulate (don't overwrite): one unsupported primitive must not
                # be masked by a later supported one.
                world_update_status = world_update_status and obj_update_status
                for cumotion_object in cumotion_objects:
                    if isinstance(cumotion_object, Cuboid):
                        cuboid_list.append(cumotion_object)
                    elif isinstance(cumotion_object, Cylinder):
                        cylinder_list.append(cumotion_object)
                    elif isinstance(cumotion_object, Sphere):
                        sphere_list.append(cumotion_object)
                    elif isinstance(cumotion_object, Mesh):
                        mesh_list.append(cumotion_object)

            world_model = WorldConfig(
                cuboid=cuboid_list,
                cylinder=cylinder_list,
                sphere=sphere_list,
                mesh=mesh_list,
            ).get_collision_check_world()
            self.motion_gen.update_world(world_model)
        if self.__read_esdf_grid:
            # AND, don't overwrite: a successful ESDF update must not mask a
            # failed PlanningScene collision object (else we'd plan as if that
            # object were absent). ESDF is the production default, so the old
            # overwrite silently dropped any unsupported MoveIt primitive.
            world_update_status = self.update_voxel_grid() and world_update_status
        if self.__publish_curobo_world_as_voxels:
            if self.__voxel_pub.get_subscription_count() > 0:
                # Calculate occupancy and publish only when subscribed.
                voxels = self.__world_collision.get_occupancy_in_bounding_box(
                    Cuboid(
                        name='test',
                        pose=[0.0, 0.0, 0.0, 1, 0, 0, 0],  # x, y, z, qw, qx, qy, qz
                        dims=self.__grid_size_m,
                    ),
                    voxel_size=self.__publish_voxel_size,
                )
                xyzr_tensor = voxels.xyzr_tensor.clone()
                xyzr_tensor[..., 3] = voxels.feature_tensor
                self.publish_voxels(xyzr_tensor)
        return world_update_status

    def execute_callback(self, goal_handle):
        if self.planner_busy:
            self.get_logger().error('Planner is busy')
            goal_handle.abort()
            result = MoveGroup.Result()
            result.error_code.val = MoveItErrorCodes.FAILURE
            return result

        self._clear_traj_viz()
        self.get_logger().info('Executing goal...')

        # check moveit scaling factors:
        min_scaling_factor = min(goal_handle.request.request.max_velocity_scaling_factor,
                                 goal_handle.request.request.max_acceleration_scaling_factor)
        time_dilation_factor = min(1.0, min_scaling_factor)

        if time_dilation_factor <= 0.0 or self.__override_moveit_scaling_factors:
            time_dilation_factor = self.get_parameter(
                'time_dilation_factor').get_parameter_value().double_value
        self.get_logger().info('Planning with time_dilation_factor: ' +
                               str(time_dilation_factor))
        plan_req = goal_handle.request.request

        # Acquire the (non-reentrant) MotionGen BEFORE any motion_gen access
        # (update_world_objects / get_active_js / reset / plan_*), so the whole
        # operation is atomic with respect to a concurrent goalset plan. The
        # acquire is before succeed() so a busy-reject aborts a goal that was
        # never succeeded (no abort-after-succeed warning).
        if not self._try_acquire_planner():
            self.get_logger().error('Planner is busy')
            goal_handle.abort()
            result = MoveGroup.Result()
            result.error_code.val = MoveItErrorCodes.FAILURE
            return result

        goal_handle.succeed()
        try:
            return self._run_execute(goal_handle, plan_req, time_dilation_factor)
        finally:
            # Always release, even if any motion_gen call raises (BUG A).
            self._release_planner()

    def _run_execute(self, goal_handle, plan_req, time_dilation_factor):
        scene = goal_handle.request.planning_options.planning_scene_diff

        world_objects = scene.world.collision_objects
        world_update_status = self.update_world_objects(world_objects)
        result = MoveGroup.Result()

        if not world_update_status:
            result.error_code.val = MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE
            self.get_logger().error('World update failed.')
            return result
        start_state = None
        if len(plan_req.start_state.joint_state.position) > 0:
            start_state = self.motion_gen.get_active_js(
                CuJointState.from_position(
                    position=self.tensor_args.to_device(
                        plan_req.start_state.joint_state.position
                    ).unsqueeze(0),
                    joint_names=plan_req.start_state.joint_state.name,
                )
            )
        else:
            self.get_logger().info(
                'PlanRequest start state was empty, reading current joint state'
            )
        if start_state is None or plan_req.start_state.is_diff:
            if self.__js_buffer is None:
                self.get_logger().error(
                    'joint_state was not received from ' + self.__joint_states_topic
                )
                return result

            # read joint state:
            state = CuJointState.from_position(
                position=self.tensor_args.to_device(self.__js_buffer['position']).unsqueeze(0),
                joint_names=self.__js_buffer['joint_names'],
            )
            state.velocity = self.tensor_args.to_device(self.__js_buffer['velocity']).unsqueeze(0)
            if state.velocity.shape != state.position.shape:
                self.get_logger().error(
                    'start joint position shape is  ' + str(state.position.shape) +
                    ' start velocity shape is ' + str(state.velocity.shape) +
                    ', both should match. JointState was read from ' + self.__joint_states_topic
                )
                return result
            current_joint_state = self.motion_gen.get_active_js(state)
            if start_state is not None and plan_req.start_state.is_diff:
                start_state.position += current_joint_state.position
                start_state.velocity += current_joint_state.velocity
            else:
                start_state = current_joint_state

        plan_mode = "pose"
        goal_state = None
        goal_pose = None

        if len(plan_req.goal_constraints[0].joint_constraints) > 0:
            self.get_logger().info('Calculating goal pose from Joint target')
            
            # Get all joint constraints from the request
            all_goal_config = [
                plan_req.goal_constraints[0].joint_constraints[x].position
                for x in range(len(plan_req.goal_constraints[0].joint_constraints))
            ]
            all_goal_jnames = [
                plan_req.goal_constraints[0].joint_constraints[x].joint_name
                for x in range(len(plan_req.goal_constraints[0].joint_constraints))
            ]
            
            # Filter to only include active joints known to cumotion
            active_joint_names = self.motion_gen.kinematics.joint_names
            goal_config = []
            goal_jnames = []
            for jname, jpos in zip(all_goal_jnames, all_goal_config):
                if jname in active_joint_names:
                    goal_jnames.append(jname)
                    goal_config.append(jpos)
            
            self.get_logger().info(f'Filtered {len(goal_jnames)}/{len(all_goal_jnames)} joints for cumotion planning')

            if len(goal_jnames) == 0:
                self.get_logger().error('No joint constraints map to active cuMotion joints')
                result.error_code.val = MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS
                return result

            goal_state = self.motion_gen.get_active_js(
                CuJointState.from_position(
                    position=self.tensor_args.to_device(goal_config).view(1, -1),
                    joint_names=goal_jnames,
                )
            )
            plan_mode = "joint"
        elif (
            len(plan_req.goal_constraints[0].position_constraints) > 0
            and len(plan_req.goal_constraints[0].orientation_constraints) > 0
        ):
            self.get_logger().info('Using goal from Pose')

            position = (
                plan_req.goal_constraints[0]
                .position_constraints[0]
                .constraint_region.primitive_poses[0]
                .position
            )
            position = [position.x, position.y, position.z]
            orientation = plan_req.goal_constraints[0].orientation_constraints[0].orientation
            orientation = [orientation.w, orientation.x, orientation.y, orientation.z]
            
            # Log target pose for debugging
            self.get_logger().info(f'Target Position: x={position[0]:.4f}, y={position[1]:.4f}, z={position[2]:.4f}')
            self.get_logger().info(f'Target Orientation (wxyz): w={orientation[0]:.4f}, x={orientation[1]:.4f}, y={orientation[2]:.4f}, z={orientation[3]:.4f}')
            
            pose_list = position + orientation
            goal_pose = Pose.from_list(pose_list, tensor_args=self.tensor_args)

            # Check if link names match:
            position_link_name = plan_req.goal_constraints[0].position_constraints[0].link_name
            orientation_link_name = (
                plan_req.goal_constraints[0].orientation_constraints[0].link_name
            )
            plan_link_name = self.motion_gen.kinematics.ee_link
            if position_link_name != orientation_link_name:
                self.get_logger().error(
                    'Link name for Target Position "'
                    + position_link_name
                    + '" and Target Orientation "'
                    + orientation_link_name
                    + '" do not match'
                )
                result.error_code.val = MoveItErrorCodes.INVALID_LINK_NAME
                return result
            if position_link_name != plan_link_name:
                self.get_logger().error(
                    'Link name for Target Pose "'
                    + position_link_name
                    + '" and Planning frame "'
                    + plan_link_name
                    + '" do not match, relaunch node with tool_frame = '
                    + position_link_name
                )
                result.error_code.val = MoveItErrorCodes.INVALID_LINK_NAME
                return result
        else:
            self.get_logger().error('Goal constraints not supported')
            result = MoveGroup.Result()
            result.error_code.val = MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS
            return result
        # Planner already reserved at the top of execute_callback (before
        # update_world_objects) and released in its finally; reset()+plan run
        # plain here inside the protected _run_execute body.
        self.motion_gen.reset(reset_seed=False)
        if plan_mode == "joint":
            motion_gen_result = self.motion_gen.plan_single_js(
                start_state,
                goal_state,
                MotionGenPlanConfig(
                    max_attempts=self.__max_attempts,
                    enable_graph_attempt=1,
                    time_dilation_factor=time_dilation_factor,
                ),
            )
        else:
            motion_gen_result = self.motion_gen.plan_single(
                start_state,
                goal_pose,
                MotionGenPlanConfig(
                    max_attempts=self.__max_attempts,
                    enable_graph_attempt=1,
                    time_dilation_factor=time_dilation_factor,
                ),
            )
        # One-shot snapshot after the first real plan — captures any allocation
        # the goal-set plan adds on top of warmup (CUDA graphs for the actual
        # problem shapes). No-op unless CUMOTION_MEM_SNAPSHOT is set.
        if not self.__mem_snap_after_plan_done:
            self.__mem_snap_after_plan_done = True
            _dump_mem_snapshot(self, '/tmp/cumotion_mem_after_first_plan.pickle')
        result = MoveGroup.Result()
        if motion_gen_result.success.item():
            result.error_code.val = MoveItErrorCodes.SUCCESS
            result.trajectory_start = plan_req.start_state
            traj = self.get_joint_trajectory(
                motion_gen_result.optimized_plan, motion_gen_result.optimized_dt.item()
            )
            result.planning_time = motion_gen_result.total_time
            result.planned_trajectory = traj
            try:
                self._publish_trajectory_visualization(motion_gen_result)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f'trajectory viz publish failed: {exc}'
                )
        elif not motion_gen_result.valid_query:
            self.get_logger().error(
                f'Invalid planning query: {motion_gen_result.status}'
            )
            if motion_gen_result.status == MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS:
                result.error_code.val = MoveItErrorCodes.START_STATE_INVALID
            if motion_gen_result.status in [
                    MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION,
                    MotionGenStatus.INVALID_START_STATE_SELF_COLLISION,
            ]:

                result.error_code.val = MoveItErrorCodes.START_STATE_IN_COLLISION
        else:
            self.get_logger().error(
                f'Motion planning failed wih status: {motion_gen_result.status}'
            )
            if motion_gen_result.status == MotionGenStatus.IK_FAIL:
                result.error_code.val = MoveItErrorCodes.NO_IK_SOLUTION
            try:
                self._publish_trajectory_visualization(
                    motion_gen_result, success=False
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f'trajectory viz (failed plan) publish failed: {exc}'
                )

        self.get_logger().info(
            'returned planning result (query, success, failure_status): '
            + str(self.__query_count)
            + ' '
            + str(motion_gen_result.success.item())
            + ' '
            + str(motion_gen_result.status)
        )
        self.__query_count += 1
        return result

    def _tcp_points_from_q(self, q_tensor):
        # q_tensor: [..., dof] torch tensor on planner device
        flat = q_tensor.reshape(-1, q_tensor.shape[-1])
        flat = flat.to(dtype=self.tensor_args.dtype, device=self.tensor_args.device)
        state = self.motion_gen.kinematics.get_state(flat)
        ee = state.ee_position.detach().cpu().numpy()
        pts = []
        for i in range(ee.shape[0]):
            p = Point()
            p.x = float(ee[i, 0])
            p.y = float(ee[i, 1])
            p.z = float(ee[i, 2])
            pts.append(p)
        return pts

    def _make_traj_marker(self, ns, marker_id, points, rgb, width=0.005):
        marker = Marker()
        frame = (
            self._traj_viz_frame_override
            if self._traj_viz_frame_override
            else self._CumotionActionServer__robot_base_frame
        )
        marker.header.frame_id = frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = width
        marker.pose.orientation.w = 1.0
        marker.color.r = float(rgb[0])
        marker.color.g = float(rgb[1])
        marker.color.b = float(rgb[2])
        marker.color.a = 1.0
        marker.points = points
        return marker

    def _clear_traj_viz(self):
        # Called at the start of every plan: stop any in-flight animation from
        # the previous plan and tell rviz to drop every marker on the viz
        # topic (DELETEALL covers all namespaces, so iter overlays + final
        # marker are cleared in one shot).
        with self._traj_viz_lock:
            self._traj_viz_anim_stop.set()
            if (
                self._traj_viz_anim_thread is not None
                and self._traj_viz_anim_thread.is_alive()
            ):
                self._traj_viz_anim_thread.join(timeout=0.1)
            self._traj_viz_anim_thread = None
            self._traj_viz_anim_stop = threading.Event()
        self._traj_viz_iter_count = 0
        m = Marker()
        m.header.frame_id = self._CumotionActionServer__robot_base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.action = Marker.DELETEALL
        self._traj_viz_pub.publish(m)

    def _publish_trajectory_visualization(self, motion_gen_result, success=True):
        # Pre-compute everything on the planner thread (we need access to
        # motion_gen.kinematics for FK), then hand off to a worker that
        # staggers the publishes so rviz sees them appear progressively.

        iter_pts_list = []
        if self._publish_iter_trajs:
            debug = getattr(motion_gen_result, 'debug_info', None)
            iters = self._extract_iter_q_list(debug)
            if iters:
                stride = self._iter_publish_stride
                selected = iters[::stride]
                if selected and selected[-1] is not iters[-1]:
                    selected.append(iters[-1])
                for q_iter in selected:
                    if q_iter.dim() == 3:
                        q_iter = q_iter[0]
                    pts = self._tcp_points_from_q(q_iter)
                    if len(pts) >= 2:
                        iter_pts_list.append(pts)

        opt_plan = getattr(motion_gen_result, 'optimized_plan', None)
        final_pts = None
        if opt_plan is not None and opt_plan.position is not None:
            q = opt_plan.position
            if q.dim() == 3:
                q = q[0]
            pts = self._tcp_points_from_q(q)
            if len(pts) >= 2:
                final_pts = pts
        else:
            self.get_logger().warn(
                'traj viz: motion_gen_result has no optimized_plan'
            )

        prev_iter = self._traj_viz_iter_count
        iter_n = len(iter_pts_list)
        self._traj_viz_iter_count = iter_n
        rgb_final = (0.0, 1.0, 0.0) if success else (1.0, 0.0, 0.0)
        self.get_logger().info(
            f'traj viz: {"SUCCESS" if success else "FAILED"} '
            f'final={"yes" if final_pts is not None else "no"} '
            f'iters={iter_n} (animation {self._iter_animation_period:.2f}s)'
        )

        # Stop any in-flight animation from a previous plan.
        with self._traj_viz_lock:
            self._traj_viz_anim_stop.set()
            if (
                self._traj_viz_anim_thread is not None
                and self._traj_viz_anim_thread.is_alive()
            ):
                self._traj_viz_anim_thread.join(timeout=0.1)
            self._traj_viz_anim_stop = threading.Event()
            stop_evt = self._traj_viz_anim_stop
            self._traj_viz_anim_thread = threading.Thread(
                target=self._animate_traj_viz,
                args=(iter_pts_list, prev_iter, final_pts, rgb_final, stop_evt),
                daemon=True,
            )
            self._traj_viz_anim_thread.start()

    def _animate_traj_viz(
        self, iter_pts_list, prev_iter, final_pts, rgb_final, stop_evt
    ):
        n = len(iter_pts_list)
        period = self._iter_animation_period
        dt = (period / n) if n > 0 and period > 0 else 0.0
        for idx, pts in enumerate(iter_pts_list):
            if stop_evt.is_set():
                return
            self._traj_viz_pub.publish(
                self._make_traj_marker(
                    'cumotion_iter', idx, pts,
                    (1.0, 1.0, 0.0),  # yellow = optimization process
                    width=0.003,
                )
            )
            if dt > 0.0 and idx < n - 1:
                stop_evt.wait(dt)

        # Delete leftover iter markers from a previous longer plan.
        if prev_iter > n:
            for i in range(n, prev_iter):
                if stop_evt.is_set():
                    return
                m = Marker()
                m.header.frame_id = self._CumotionActionServer__robot_base_frame
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = 'cumotion_iter'
                m.id = i
                m.action = Marker.DELETE
                self._traj_viz_pub.publish(m)

        # Final trajectory drawn last (green=success, red=failure).
        if final_pts is not None and not stop_evt.is_set():
            self._traj_viz_pub.publish(
                self._make_traj_marker(
                    'cumotion_planned', 0, final_pts, rgb_final, width=0.006,
                )
            )

    def _extract_iter_q_list(self, debug):
        # Pull per-iter best_q tensors out of curobo's nested debug structure.
        # Real shape (curobo 0.7.x):
        #   motion_gen_result.debug_info
        #     = {"trajopt_result": TrajOptResult}
        #   TrajOptResult.debug_info
        #     = {"solver": WrapResult.debug, ...}
        #   WrapResult.debug
        #     = {"steps": [particle_opt.debug, newton_opt.debug], "cost": [...]}
        #   newton_opt.debug
        #     = [tensor[bs*seeds, horizon, dof], tensor, ...]   <- what we want
        #
        # The walker must avoid sweeping in unrelated tensors that *also* live
        # on TrajOptResult (e.g. `seed`, `solution.position`, `raw_solution`,
        # `metrics.*`, `goal.*`) — those are full-batch tensors of every seed,
        # and turning them into LINE_STRIPs paints "random" yellow lines.
        # We accept ONLY lists/tuples of trajectory-shaped tensors:
        #   tensor.shape[-1] == dof  AND  tensor.shape[-2] >= 2
        # which is exactly the convention NewtonOptBase uses for `self.debug`.
        out = []
        if debug is None:
            self.get_logger().warn('traj viz: debug_info is None')
            return out

        try:
            dof = len(self.motion_gen.kinematics.joint_names)
        except Exception:  # noqa: BLE001
            dof = None

        def is_traj_tensor(t):
            return (
                torch.is_tensor(t)
                and t.dim() >= 2
                and t.shape[-2] >= 2
                and (dof is None or t.shape[-1] == dof)
            )

        seen_lists = 0
        seen_dicts = 0
        seen_objs = 0
        visited_ids = set()

        def visit(obj):
            nonlocal seen_lists, seen_dicts, seen_objs
            if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
                return
            oid = id(obj)
            if oid in visited_ids:
                return
            visited_ids.add(oid)
            if isinstance(obj, (list, tuple)):
                seen_lists += 1
                # Only treat this list as a per-iter snapshot list if every
                # element is a trajectory-shaped tensor. Otherwise descend.
                if obj and all(is_traj_tensor(x) for x in obj):
                    out.extend(obj)
                else:
                    for item in obj:
                        visit(item)
            elif isinstance(obj, dict):
                seen_dicts += 1
                for v in obj.values():
                    visit(v)
            elif torch.is_tensor(obj):
                # Don't unwrap bare tensors — they are usually full-batch
                # solutions/seeds, not per-iter snapshots.
                return
            elif hasattr(obj, '__dataclass_fields__'):
                seen_objs += 1
                # For dataclasses (e.g. TrajOptResult) descend ONLY through
                # `debug_info`. The other fields hold final/seed/metric tensors
                # that aren't iteration progress.
                if hasattr(obj, 'debug_info'):
                    visit(getattr(obj, 'debug_info'))
            elif hasattr(obj, '__dict__'):
                seen_objs += 1
                if 'debug_info' in vars(obj):
                    visit(vars(obj)['debug_info'])
                elif 'debug' in vars(obj):
                    visit(vars(obj)['debug'])

        visit(debug)
        self.get_logger().info(
            f'traj viz: extracted {len(out)} iter snapshots '
            f'(visited {seen_lists} lists, {seen_dicts} dicts, {seen_objs} objs)'
        )
        return out

    def publish_voxels(self, voxels):
        vox_size = self.__publish_voxel_size

        # create marker:
        marker = Marker()
        marker.header.frame_id = self.__robot_base_frame
        marker.id = 0
        marker.type = 6  # cube list
        marker.ns = 'curobo_world'
        marker.action = 0
        marker.pose.orientation.w = 1.0
        marker.lifetime = rclpy.duration.Duration(seconds=0.0).to_msg()
        marker.frame_locked = False
        marker.scale.x = vox_size
        marker.scale.y = vox_size
        marker.scale.z = vox_size
        marker.points = []

        # get only voxels that are inside surfaces:
        voxels = voxels[voxels[:, 3] > 0.0]
        vox = voxels.view(-1, 4).cpu().numpy()
        number_of_voxels_to_publish = len(vox)
        if len(vox) > self.__max_publish_voxels:
            self.get_logger().warn(
                f'Number of voxels to publish bigger than max_publish_voxels, '
                f'{len(vox)} > {self.__max_publish_voxels}'
            )
            number_of_voxels_to_publish = self.__max_publish_voxels
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        vox = vox.astype(np.float64)
        for i in range(number_of_voxels_to_publish):
            # Publish the markers at the center of the voxels:
            pt = Point()
            pt.x = vox[i, 0]
            pt.y = vox[i, 1]
            pt.z = vox[i, 2]
            marker.points.append(pt)

        # publish voxels:
        marker.header.stamp = self.get_clock().now().to_msg()

        self.__voxel_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    cumotion_action_server = CumotionActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(cumotion_action_server)
    try:
        executor.spin()
    except KeyboardInterrupt:
        cumotion_action_server.get_logger().info('KeyboardInterrupt, shutting down.\n')
    cumotion_action_server.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
