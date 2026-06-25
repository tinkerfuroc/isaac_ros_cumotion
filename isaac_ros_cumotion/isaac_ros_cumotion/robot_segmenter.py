# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from copy import deepcopy
import threading
import time

from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose as CuPose
from curobo.types.state import JointState as CuJointState
from curobo.wrap.model.robot_segmenter import RobotSegmenter
import cv2
from cv_bridge import CvBridge
from isaac_ros_common.qos import add_qos_parameter
from isaac_ros_cumotion.update_kinematics import get_robot_config
from isaac_ros_cumotion.update_kinematics import UpdateLinkSpheresServer
from isaac_ros_cumotion.util import get_spheres_marker
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import torch
from visualization_msgs.msg import MarkerArray


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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


class CumotionRobotSegmenter(Node):
    """This node filters out depth pixels assosiated with a robot body using a mask."""

    def __init__(self):
        super().__init__('cumotion_robot_segmentation')
        _mem_report(self, '00_init_start')
        self.declare_parameter('robot', 'ur5e.yml')
        self.declare_parameter('urdf_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('yml_file_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('cuda_device', 0)
        self.declare_parameter('distance_threshold', 0.1)
        # use_cuda_graph captures the masking kernels into a CUDA graph for speed,
        # but that pins a multi-GB pool sized for the full-res depth peak (measured
        # ~3.9GB) that the caching allocator can never release. Set False to trade
        # a little per-frame latency for that VRAM. See scripts/ GPU profiling.
        self.declare_parameter('use_cuda_graph', True)
        self.declare_parameter('time_sync_slop', 0.1)
        self.declare_parameter('tf_lookup_duration', 5.0)

        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('debug_robot_topic', '/cumotion/robot_segmenter/robot_spheres')

        self.declare_parameter('depth_image_topics', ['/cumotion/depth_1/image_raw'])
        self.declare_parameter('depth_camera_infos', ['/cumotion/depth_1/camera_info'])
        self.declare_parameter('robot_mask_publish_topics', ['/cumotion/depth_1/robot_mask'])
        self.declare_parameter('world_depth_publish_topics', ['/cumotion/depth_1/world_depth'])

        self.declare_parameter('filter_speckles_in_mask', False)
        self.declare_parameter('max_filtered_speckles_size', 1250)

        self.declare_parameter('log_debug', False)
        self.declare_parameter('update_link_sphere_server',
                               'segmenter_attach_object')

        depth_qos = add_qos_parameter(self, 'DEFAULT', 'depth_qos')
        depth_info_qos = add_qos_parameter(self, 'DEFAULT', 'depth_info_qos')
        mask_qos = add_qos_parameter(self, 'DEFAULT', 'mask_qos')
        world_depth_qos = add_qos_parameter(self, 'DEFAULT', 'world_depth_qos')

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

        distance_threshold = (
            self.get_parameter('distance_threshold').get_parameter_value().double_value)
        time_sync_slop = self.get_parameter('time_sync_slop').get_parameter_value().double_value
        self._tf_lookup_duration = (
            self.get_parameter('tf_lookup_duration').get_parameter_value().double_value
        )
        joint_states_topic = (
            self.get_parameter('joint_states_topic').get_parameter_value().string_value)
        debug_robot_topic = (
            self.get_parameter('debug_robot_topic').get_parameter_value().string_value)
        depth_image_topics = (
            self.get_parameter('depth_image_topics').get_parameter_value().string_array_value)
        depth_camera_infos = (
            self.get_parameter('depth_camera_infos').get_parameter_value().string_array_value)
        publish_mask_topics = (
            self.get_parameter(
                'robot_mask_publish_topics').get_parameter_value().string_array_value)
        world_depth_topics = (
            self.get_parameter(
                'world_depth_publish_topics').get_parameter_value().string_array_value)
        self._filter_speckles_in_mask = (
            self.get_parameter('filter_speckles_in_mask').get_parameter_value().bool_value
        )
        self._max_filtered_speckles_size = self.get_parameter(
            'max_filtered_speckles_size').get_parameter_value().integer_value
        self._update_link_sphere_server = (
            self.get_parameter('update_link_sphere_server').get_parameter_value().string_value)

        self._log_debug = self.get_parameter('log_debug').get_parameter_value().bool_value
        num_cameras = len(depth_image_topics)
        self._num_cameras = num_cameras

        if len(depth_camera_infos) != num_cameras:
            self.get_logger().error(
                'Number of topics in depth_camera_infos does not match depth_image_topics')
        if len(publish_mask_topics) != num_cameras:
            self.get_logger().error(
                'Number of topics in publish_mask_topics does not match depth_image_topics')
        if len(world_depth_topics) != num_cameras:
            self.get_logger().error(
                'Number of topics in world_depth_topics does not match depth_image_topics')

        cuda_device_id = self.get_parameter('cuda_device').get_parameter_value().integer_value

        self._tensor_args = TensorDeviceType(device=torch.device('cuda', cuda_device_id))

        # Create subscribers:
        #
        # PER-CAMERA synchronization. Previously a single ApproximateTimeSynchronizer
        # spanned ALL depth subscribers + joint_states and emitted one batched
        # callback, which forced every camera to the SAME rate (slowest wins) and
        # the SAME resolution (the old on_timer np.stack'd them into one tensor).
        # The wrist FFS (~3 Hz, 426x240) and the head Orbbec (~30 Hz, 212x192) are
        # incompatible under that scheme. Now each camera gets its own
        # (depth_i, joint_states) synchronizer so cameras run independently at
        # their native rate/resolution while still sharing ONE curobo robot model.
        self._depth_subscribers = []
        self._joint_subscribers = []
        self.approx_time_syncs = []
        for idx in range(num_cameras):
            depth_sub = Subscriber(self, Image, depth_image_topics[idx], qos_profile=depth_qos)
            joint_sub = Subscriber(self, JointState, joint_states_topic)
            sync = ApproximateTimeSynchronizer(
                (depth_sub, joint_sub), queue_size=10, slop=time_sync_slop)
            # Bind idx via default arg so each lambda captures its own camera index.
            sync.registerCallback(
                lambda depth_msg, js_msg, index=idx:
                    self.process_depth_and_joint_state(depth_msg, js_msg, index))
            self._depth_subscribers.append(depth_sub)
            self._joint_subscribers.append(joint_sub)
            self.approx_time_syncs.append(sync)

        self.info_subscribers = []

        for idx in range(num_cameras):
            self.info_subscribers.append(
                self.create_subscription(
                    CameraInfo, depth_camera_infos[idx],
                    lambda msg, index=idx: self.camera_info_cb(msg, index), depth_info_qos)
            )

        self.mask_publishers = [
            self.create_publisher(Image, topic, mask_qos) for topic in publish_mask_topics]
        self.segmented_publishers = [
            self.create_publisher(Image, topic, world_depth_qos) for topic in world_depth_topics]

        self.debug_robot_publisher = self.create_publisher(MarkerArray, debug_robot_topic, 10)

        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=60.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.br = CvBridge()

        # Create buffers to store data:
        #
        # All depth-side buffers are now PER-CAMERA indexed lists (the old code
        # rebuilt them as parallel lists inside one batched callback). A camera
        # whose entry is still None simply hasn't produced a synced frame yet and
        # is skipped this tick. The joint-state buffer is shared (latest from any
        # camera's callback) and a snapshot is taken once per on_timer tick so all
        # cameras in a tick use the same robot pose.
        self._depth_buffers = [None for x in range(num_cameras)]
        self._depth_intrinsics = [None for x in range(num_cameras)]
        self._robot_pose_camera = [None for x in range(num_cameras)]
        self._depth_encoding = [None for x in range(num_cameras)]
        self._camera_headers = [None for x in range(num_cameras)]
        # Per-camera stamp of the depth frame most recently MASKED, used to skip
        # re-masking the same frame on every 100 Hz timer tick (the old single-
        # batch design cleared its buffers after each masked tick to the same end;
        # here each camera is independent so we dedup per camera by frame stamp).
        self._processed_stamp = [None for x in range(num_cameras)]

        self._js_buffer = None
        self._timestamp = None
        # PER-CAMERA joint state (FIX 6 — the correctness fix). Each camera is
        # synchronized against /joint_states by its OWN ApproximateTimeSynchronizer,
        # so the joint state paired with camera i's depth is generally NOT the same
        # as the shared self._js_buffer (which any camera's callback overwrites).
        # These hold the joint state (and its header.stamp) that camera i's most
        # recent depth was actually synced against, so on_timer can mask camera i
        # with the matching robot pose AND look up its extrinsic at the matching
        # joint-state stamp. The shared self._js_buffer/self._timestamp are KEPT
        # for the readiness gate and the debug link-spheres publish only.
        self._cam_js = [None for x in range(num_cameras)]
        self._cam_js_stamp = [None for x in range(num_cameras)]
        # Tracks the (height, width) the shared RobotSegmenter's cached projection
        # rays / scratch buffers are currently sized for. When the next camera in
        # the loop has a different resolution we reset those caches before
        # re-projecting (see on_timer). Stays constant for the single-camera case,
        # so that path keeps reusing the cached buffers exactly as before.
        self._seg_cached_hw = None
        # Index of the camera the segmenter's projection rays were last built for.
        # Projection rays bake in per-camera intrinsics, so we must also re-project
        # when the processed camera index changes (not only on a resolution change).
        # For the single-camera case idx is always 0, so this never forces an extra
        # re-projection after the first frame.
        self._last_proj_cam_idx = None
        self.lock = threading.Lock()
        self.timer = self.create_timer(0.01, self.on_timer)

        robot_config = get_robot_config(
            robot_file=self.__robot_file,
            urdf_file_path=self.__urdf_path,
            logger=self.get_logger()
        )

        seg_use_cuda_graph = self.get_parameter(
            'use_cuda_graph').get_parameter_value().bool_value
        self._cumotion_segmenter = RobotSegmenter.from_robot_file(
            robot_config, distance_threshold=distance_threshold,
            use_cuda_graph=seg_use_cuda_graph)
        self.get_logger().info(
            f'RobotSegmenter use_cuda_graph={seg_use_cuda_graph}')
        _mem_report(self, '01_after_robot_world_load')

        self._cumotion_base_frame = self._cumotion_segmenter.base_link

        self.__update_link_spheres_server = UpdateLinkSpheresServer(
            server_node=self,
            action_name=self._update_link_sphere_server,
            robot_kinematics=self._cumotion_segmenter.robot_world.kinematics,
            robot_base_frame=self._cumotion_base_frame
        )

        self.get_logger().info(f'Node initialized with {self._num_cameras} cameras')
        _mem_report(self, '02_init_done')
        self.create_timer(5.0, lambda: _mem_report(self, 'periodic'))

    def process_depth_and_joint_state(self, depth_msg, js_msg, camera_idx):
        img = self.br.imgmsg_to_cv2(depth_msg)
        if depth_msg.encoding == '32FC1':
            img = 1000.0 * img
        # Only the fast buffer copies are done under the lock; no GPU work here.
        with self.lock:
            self._depth_buffers[camera_idx] = img
            self._camera_headers[camera_idx] = depth_msg.header
            self._depth_encoding[camera_idx] = depth_msg.encoding
            # FIX 6: record the joint state THIS camera's depth was synced against,
            # plus its stamp, so on_timer masks camera_idx with the matching pose
            # and looks up its extrinsic at the matching joint-state time. Stored
            # by REPLACING the slot (new dict / new stamp object), never mutating in
            # place, so on_timer's under-lock list() snapshot is a stable view.
            self._cam_js[camera_idx] = {
                'joint_names': js_msg.name, 'position': js_msg.position}
            self._cam_js_stamp[camera_idx] = js_msg.header.stamp
            # Shared latest joint state (KEPT for the readiness gate + debug spheres).
            self._js_buffer = {'joint_names': js_msg.name, 'position': js_msg.position}
            self._timestamp = js_msg.header.stamp

    def camera_info_cb(self, msg, idx):
        # FIX 7 (hardening): guard the write for symmetry with on_timer, which
        # reads self._depth_intrinsics under self.lock. Single-threaded executor
        # today, but this removes the only asymmetric unguarded shared write.
        with self.lock:
            self._depth_intrinsics[idx] = msg.k

    def publish_robot_spheres(self, traj: CuJointState):
        kin_state = self._cumotion_segmenter.robot_world.get_kinematics(traj.position)
        spheres = kin_state.link_spheres_tensor.cpu().numpy()
        current_time = self.get_clock().now().to_msg()

        m_arr = get_spheres_marker(
            spheres[0],
            self._cumotion_base_frame,
            current_time,
            rgb=[0.0, 1.0, 0.0, 1.0],
        )

        self.debug_robot_publisher.publish(m_arr)

    def is_subscribed(self) -> bool:
        count_mask = max(
            [mask_pub.get_subscription_count() for mask_pub in self.mask_publishers]
            + [seg_pub.get_subscription_count() for seg_pub in self.segmented_publishers]
        )
        if count_mask > 0:
            return True
        return False

    def filter_depth_mask(self, robot_mask, depth_image):
        # pixels with depth <= 0.0 are invalid
        invalid_depth_value = 0.0
        # get the invalid depth mask
        invalid_depth_mask = depth_image <= invalid_depth_value
        # combine the invalid depth and robot masks
        combined_mask = np.logical_or(robot_mask, invalid_depth_mask).astype(np.uint8) * 255
        # filter speckles from the combined mask
        filtered_combined_mask = cv2.filterSpeckles(
            combined_mask, 255, self._max_filtered_speckles_size, 0)[0]
        # Set depth pixels to invalid if they are masked in the filtered mask
        depth_image[filtered_combined_mask.astype(bool)] = invalid_depth_value
        return (filtered_combined_mask, depth_image)

    def publish_images(self, depth_mask, segmented_depth, camera_header, idx: int):
        # depth_mask / segmented_depth are this camera's single (H_i, W_i) arrays;
        # camera_header is this camera's header. idx selects the matching publisher
        # and per-camera encoding.
        if self._filter_speckles_in_mask:
            depth_mask, segmented_depth = self.filter_depth_mask(depth_mask, segmented_depth)

        if self.mask_publishers[idx].get_subscription_count() > 0:
            msg = self.br.cv2_to_imgmsg(depth_mask, 'mono8')
            msg.header = camera_header
            self.mask_publishers[idx].publish(msg)

        if self.segmented_publishers[idx].get_subscription_count() > 0:
            if self._depth_encoding[idx] == '16UC1':
                segmented_depth = segmented_depth.astype(np.uint16)
            elif self._depth_encoding[idx] == '32FC1':
                segmented_depth = segmented_depth / 1000.0
            msg = self.br.cv2_to_imgmsg(segmented_depth, self._depth_encoding[idx])
            msg.header = camera_header
            self.segmented_publishers[idx].publish(msg)

    def _reset_seg_caches_for_resolution(self):
        """Drop the shared RobotSegmenter's resolution-dependent caches.

        The shared curobo RobotSegmenter caches projection rays and scratch
        buffers sized for ONE depth resolution:
          - _projection_rays  (b, H*W, 3), refreshed in-place via .copy_()
          - _out_points_buffer / _out_gpt  ((b, H*W, 3)), allocated lazily in
            _mask_op from the first frame's point count.
        update_camera_projection() refreshes the rays with .copy_(), which
        requires a matching shape — so feeding a second camera of a DIFFERENT
        resolution into the same instance would raise. Before re-projecting for a
        camera whose (H, W) differs from the currently-cached one, null these
        caches so they are re-allocated at the new resolution on the next call.
        (_out_gp / _out_gq depend only on batch size, which is always 1 here, so
        they are left untouched.) For a single fixed-resolution camera this is
        never invoked after the first projection, so that path is unchanged.

        The CUDA-graph state is ALSO reset: a captured graph is sized for one
        depth resolution, so replaying it after a resolution switch would feed a
        wrong-shaped graph. _cu_graph is always present (set in __init__), so it
        is nulled directly; _cu_cam_obs/_cu_q/_cu_out/_cu_filtered_out are only
        created in _create_cg_graph (and thus unset when use_cuda_graph=False), so
        they are nulled only if present. With use_cuda_graph=False this teardown is
        inert (no graph is ever captured) but keeps the helper correct.
        """
        seg = self._cumotion_segmenter
        seg._projection_rays = None
        seg.ready = False
        seg._out_points_buffer = None
        seg._out_gpt = None
        # CUDA-graph caches. _cu_graph always exists (initialized in __init__);
        # the capture-time tensors only exist after a graph has been built.
        seg._cu_graph = None
        for attr in ('_cu_cam_obs', '_cu_q', '_cu_out', '_cu_filtered_out'):
            if hasattr(seg, attr):
                setattr(seg, attr, None)

    def on_timer(self):
        # CONCURRENCY INVARIANT (FIX 8): this per-camera masking loop and the
        # shared curobo RobotSegmenter GPU state it drives (projection rays,
        # _out_*/scratch buffers, captured CUDA graph) are single-owner —
        # timer-thread-only — and the loop MUST remain serial: each iteration
        # re-points the one shared segmenter at a different camera's
        # resolution/intrinsics, so two on_timer runs overlapping would corrupt
        # that state. The node runs under the default single-threaded executor, so
        # on_timer never re-enters. If this node is EVER moved to a
        # MultiThreadedExecutor, on_timer must be placed in a
        # MutuallyExclusiveCallbackGroup (and the segmenter access kept serial).
        # The under-lock snapshot below relies on the callbacks REPLACING buffer
        # slots (assigning a fresh array/dict/stamp), never mutating a stored array
        # in place — so list()/copy here yields a stable, self-consistent view.
        computation_time = -1.0
        node_time = -1.0

        if not self.is_subscribed():
            return

        # Need at least one camera's intrinsics, at least one buffered depth
        # frame, and a joint-state timestamp before any work is possible.
        if ((not any(isinstance(intrinsic, np.ndarray) for intrinsic in self._depth_intrinsics))
                or (all(h is None for h in self._camera_headers))
                or (self._timestamp is None)):
            return

        start_node_time = time.time()

        # Snapshot the shared joint state + per-camera buffers ONCE under the lock
        # so every camera processed in this tick uses the SAME robot pose. Only
        # fast CPU copies happen under the lock; all GPU work is outside it.
        with self.lock:
            if self._js_buffer is None:
                return
            js = np.copy(self._js_buffer['position'])
            j_names = deepcopy(self._js_buffer['joint_names'])
            depth_buffers = list(self._depth_buffers)
            camera_headers = list(self._camera_headers)
            depth_intrinsics = list(self._depth_intrinsics)
            # FIX 6: snapshot the per-camera joint state + its stamp. list() is a
            # shallow copy of the slot references; callbacks REPLACE slots (never
            # mutate the stored dict/stamp), so each captured entry stays the exact
            # (joint state, stamp) that camera's depth was synced against.
            cam_js = list(self._cam_js)
            cam_js_stamp = list(self._cam_js_stamp)

        # Determine which cameras have a FRESH frame to mask BEFORE doing any GPU
        # work. On an idle 100 Hz tick (no new frames since last process) this lets
        # us return early without building q (a GPU op) — restoring the original
        # "no GPU work on idle ticks" behaviour.
        pending = []
        for i in range(self._num_cameras):
            depth_np = depth_buffers[i]
            header = camera_headers[i]
            intrinsic = depth_intrinsics[i]
            # Skip a camera that has not produced depth / intrinsics / a header yet,
            # or has no synced joint state yet (FIX 6: q is built from cam_js[i]).
            if (depth_np is None or header is None
                    or not isinstance(intrinsic, np.ndarray)
                    or cam_js[i] is None):
                continue
            # Skip a frame we have already masked (avoid re-masking the same image
            # on every tick).
            frame_stamp = (header.stamp.sec, header.stamp.nanosec)
            if self._processed_stamp[i] == frame_stamp:
                continue
            pending.append(i)

        if not pending:
            return

        # FIX 6: q is now built PER CAMERA inside the loop from cam_js[i] (the joint
        # state that camera's depth was synced against), so there is no single
        # shared q build here anymore.

        start_segmentation_time = time.time()
        any_processed = False
        # Holds the most-recently-built per-camera active js, reused for the debug
        # robot-spheres publish at the end (RViz viz only).
        last_q = None

        # Process cameras in array order so camera_0 (wrist) is always first.
        for i in pending:
            depth_np = depth_buffers[i]
            header = camera_headers[i]
            intrinsic = depth_intrinsics[i]

            frame_stamp = (header.stamp.sec, header.stamp.nanosec)

            # FIX 6: build THIS camera's active joint state from the joint state its
            # depth was synced against (cam_js[i]), not a shared js overwritten by
            # other cameras' callbacks. cam_js[i] is guaranteed non-None here (the
            # pending pre-pass skipped None entries), but guard defensively.
            cam_js_i = cam_js[i]
            if cam_js_i is None:
                continue
            q_i = CuJointState.from_numpy(
                position=np.copy(cam_js_i['position']),
                joint_names=deepcopy(cam_js_i['joint_names']),
                tensor_args=self._tensor_args).unsqueeze(0)
            q_i = self._cumotion_segmenter.robot_world.get_active_js(q_i)
            last_q = q_i

            # Per-camera TF lookup (base_frame <- this camera's optical frame).
            #
            # WRIST (camera index 0): frozen extrinsic. It barely images the arm
            # and the validated single-camera behaviour caches the TF once. So for
            # idx 0 we look up only on first acquire and reuse the cached pose.
            #
            # HEAD and any camera index >= 1: mounted on the moving pan-tilt, so the
            # extrinsic changes every frame — RE-LOOKUP every tick. Never mask the
            # head with a stale extrinsic; on lookup failure skip this frame.
            need_tf = (self._robot_pose_camera[i] is None) or (i > 0)
            if need_tf:
                try:
                    # FIX 6: look up THIS camera's extrinsic at the stamp of the
                    # joint state its depth was synced against (cam_js_stamp[i]),
                    # not the shared self._timestamp (overwritten by other cameras'
                    # callbacks). For the wrist (idx 0, frozen once) this is its own
                    # first-frame js stamp — identical to the original behaviour.
                    t = self.tf_buffer.lookup_transform(
                        self._cumotion_base_frame,
                        header.frame_id,
                        cam_js_stamp[i],
                        rclpy.duration.Duration(seconds=self._tf_lookup_duration),
                    )
                    self._robot_pose_camera[i] = CuPose.from_list(
                        [
                            t.transform.translation.x,
                            t.transform.translation.y,
                            t.transform.translation.z,
                            t.transform.rotation.w,
                            t.transform.rotation.x,
                            t.transform.rotation.y,
                            t.transform.rotation.z,
                        ]
                    )
                except TransformException as ex:
                    self.get_logger().debug(
                        f'Could not transform {header.frame_id}'
                        f'to { self._cumotion_base_frame}: {ex}')
                    continue

            # Build this camera's batch-1 depth tensor (1, H_i, W_i).
            depth_image = self._tensor_args.to_device(depth_np.astype(np.float32))
            depth_image = depth_image.view(1, depth_image.shape[-2], depth_image.shape[-1])
            this_hw = (depth_image.shape[-2], depth_image.shape[-1])

            # Re-point the shared segmenter's projection at THIS camera's
            # resolution. For a single fixed-resolution camera the cache is set up
            # exactly once (first frame) and reused thereafter — identical to the
            # original behaviour. With multiple differing resolutions we reset the
            # resolution-dependent caches whenever the size changes from the
            # last-processed camera before re-projecting.
            if self._seg_cached_hw is not None and self._seg_cached_hw != this_hw:
                self._reset_seg_caches_for_resolution()
            # Re-project whenever the segmenter isn't ready, the resolution changed,
            # OR the processed camera INDEX changed. Projection rays bake in this
            # camera's fx,fy,cx,cy, so two same-resolution different-intrinsics
            # cameras must each re-project. For the single-camera case idx is always
            # 0, so after the first frame (ready, same hw, last_idx == 0) this never
            # re-projects again — once-only, exactly as before.
            if ((not self._cumotion_segmenter.ready)
                    or (self._seg_cached_hw != this_hw)
                    or (self._last_proj_cam_idx != i)):
                intrinsics = self._tensor_args.to_device(
                    np.copy(intrinsic)).view(1, 3, 3)
                cam_obs_proj = CameraObservation(
                    depth_image=depth_image, intrinsics=intrinsics)
                self._cumotion_segmenter.update_camera_projection(cam_obs_proj)
                self._seg_cached_hw = this_hw
                self._last_proj_cam_idx = i
                self.get_logger().info(
                    f'Updated Projection Matrices for camera {i} at {this_hw[1]}x{this_hw[0]}')

            # FIX 2: pass the cached pose directly. CuPose.from_list already yields
            # position (1,3) / quaternion (1,4) (batch-1), matching the pre-refactor
            # CuPose.cat([p]) shape. The old .unsqueeze(0) reassigned the cached
            # pose's tensors in place (Pose.unsqueeze mutates self and returns self),
            # so the cached wrist pose grew a leading dim every tick.
            cam_obs = CameraObservation(
                depth_image=depth_image,
                pose=self._robot_pose_camera[i])

            depth_mask, segmented_depth = \
                self._cumotion_segmenter.get_robot_mask_from_active_js(cam_obs, q_i)
            depth_mask = depth_mask.cpu().numpy().astype(np.uint8) * 255
            segmented_depth = segmented_depth.cpu().numpy()

            # batch-1: take element 0 for this camera.
            self.publish_images(depth_mask[0], segmented_depth[0], header, i)
            self._processed_stamp[i] = frame_stamp
            any_processed = True

        if not any_processed:
            return

        if self._log_debug:
            torch.cuda.synchronize()
            computation_time = time.time() - start_segmentation_time

        self.__update_link_spheres_server.publish_all_active_spheres(
            robot_joint_states=js,
            robot_joint_names=j_names,
            tensor_args=self._tensor_args,
            rgb=[1.0, 0.0, 0.0, 1.0]
        )

        # Debug robot-spheres (RViz viz only): use the last per-camera active js
        # built this tick. any_processed is True here, so at least one camera was
        # masked and last_q is set; guard defensively regardless.
        if self.debug_robot_publisher.get_subscription_count() > 0 and last_q is not None:
            self.publish_robot_spheres(last_q)
        if self._log_debug:
            node_time = time.time() - start_node_time
            self.get_logger().info(f'Node Time(ms), Computation Time(ms): {node_time * 1000.0},\
                                    {computation_time * 1000.0}')


def main(args=None):

    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    cumotion_segmenter = CumotionRobotSegmenter()

    try:
        # Spin the node so the callback function is called.
        cumotion_segmenter.get_logger().info('Starting CumotionRobotSegmenter node')
        rclpy.spin(cumotion_segmenter)
    except KeyboardInterrupt:
        cumotion_segmenter.get_logger().info('Destroying CumotionRobotSegmenter node')

    # Destroy the node explicitly
    cumotion_segmenter.destroy_node()

    # Shutdown the ROS client library for Python
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
