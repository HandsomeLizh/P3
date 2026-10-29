from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import unittest


WORKSPACE = Path(__file__).resolve().parents[2]
ROMA = WORKSPACE / "third_party" / "RoMa" / "roma_vo.py"
MAPPER = (
    WORKSPACE
    / "src"
    / "t3_semantic_mapping"
    / "t3_semantic_mapping"
    / "roma_semantic_mapper_node.py"
)
REBUILDER = MAPPER.with_name("semantic_map_rebuilder.py")
INTERFACE = (
    WORKSPACE
    / "src"
    / "t3_interfaces"
    / "msg"
    / "IncrementalSemanticMap.msg"
)
CONFIG = (
    WORKSPACE
    / "src"
    / "t3_semantic_mapping"
    / "config"
    / "semantic_map.yaml"
)
LAUNCH_OPTIONS = MAPPER.parent / "launch_options.py"
MAPPER_LAUNCHES = tuple(
    MAPPER.parents[1] / "launch" / name
    for name in (
        "mapper_only.launch.py",
        "practice_live_precomputed.launch.py",
        "practice_live_online_semantics.launch.py",
    )
)
BACKUP_ROOT = WORKSPACE.parents[1] / "backup"
BACKUP_MAPPER = (
    BACKUP_ROOT
    / "t3_gridmap_v2_ws"
    / "src"
    / "t3_semantic_mapping"
    / "t3_semantic_mapping"
    / "roma_semantic_mapper_node.py"
)


def _class(tree: ast.AST, name: str) -> ast.ClassDef:
    return next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.ClassDef) and item.name == name
    )


def _method_source(source: str, tree: ast.AST, class_name: str, name: str) -> str:
    class_node = _class(tree, class_name)
    method = next(
        item
        for item in class_node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    return ast.get_source_segment(source, method) or ""


def _declared_parameters(path: Path) -> dict[str, str | None]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "declare_parameter"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            continue
        default = (
            ast.dump(node.args[1], include_attributes=False)
            if len(node.args) > 1
            else None
        )
        result[str(node.args[0].value)] = default
    return result


def _ros_endpoints(path: Path) -> list[tuple[str, str, str]]:
    """Return (kind, message type, topic expression) per ROS endpoint.

    QoS 与回调实参不参与比较：契约要保护的是消息类型和话题来源；
    QoS 变量名属于实现细节（原字符串级比较连 QoS 数值都从未覆盖）。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    endpoints = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr
            in {"create_publisher", "create_subscription", "Subscriber"}
        ):
            continue
        # message_filters.Subscriber 的首个位置参数是节点自身。
        offset = 1 if node.func.attr == "Subscriber" else 0
        message = ast.dump(node.args[offset], include_attributes=False)
        topic = ast.dump(node.args[offset + 1], include_attributes=False)
        endpoints.append((node.func.attr, message, topic))
    return sorted(endpoints)


def _config_parameter_values(path: Path) -> dict[str, str]:
    """Parse flat ``key: value`` pairs from a ROS 2 parameter YAML."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if ":" not in stripped:
            continue
        name, _, value = stripped.partition(":")
        value = value.strip()
        if value:
            values[name.strip()] = value
    return values


# FIX_04/05 之后节点新增了 LiDAR 建图、外部 LIO 定位、局部因子图融合与
# 瓦片化全局地图、无损观测存档、纯 VO 恢复后地图准入，以及
# 可验证的标准消息输入传输策略、同步回调完成回执，以及
# 默认关闭的 MPRF/激光几何复核/全局因子图闭环层。这些参数族
# 是备案的合法新增；其余任何新参数仍视为契约违规，防止未察觉的
# 接口漂移。
_ALLOWED_NEW_PARAMETER_PREFIXES = (
    "lidar_",
    "lio_",
    "fusion_",
    "pointcloud_assist_",
    "map_admission_",
    "map_observation_",
    "odometry_",
    "global_loop_",
    "global_graph_",
    "mprf_",
)
_ALLOWED_NEW_PARAMETERS = {
    "localization_mode",
    "enable_gp_prior",
    "enable_lidar_mapping",
    "enable_global_loop_closure",
    "incremental_publish_mode",
    "incremental_map_batch_topic",
    "incremental_map_replay_service",
    "map_tile_cells",
    "map_id",
    "free_class_ids",
    "occupied_class_ids",
    "publish_unchanged_map_outputs",
    "publish_layer_images",
    "publish_map_pointcloud",
    "path_publish_stride",
    "stereo_map_max_depth_m",
    "stereo_input_reliability",
    "semantic_input_reliability",
    "imu_input_reliability",
    "image_input_qos_depth",
    "imu_input_qos_depth",
    "image_sync_mode",
    "image_sync_queue_size",
    "image_sync_slop_sec",
    "stereo_sync_max_skew_sec",
    "input_processed_topic",
}
# 备案的新增订阅只允许来自这两个话题参数。
_ALLOWED_NEW_TOPIC_PARAMETERS = {"lidar_topic", "lio_odometry_topic"}
_ALLOWED_NEW_PUBLISHERS = {
    (
        "Name(id='SemanticMapBatch', ctx=Load())",
        "incremental_map_batch_topic",
    ),
    (
        "Name(id='Header', ctx=Load())",
        "input_processed_topic",
    ),
}


class GridMapRebuildContractStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roma_source = ROMA.read_text(encoding="utf-8")
        cls.roma_tree = ast.parse(cls.roma_source)
        cls.mapper_source = MAPPER.read_text(encoding="utf-8")
        cls.mapper_tree = ast.parse(cls.mapper_source)
        cls.rebuilder_source = REBUILDER.read_text(encoding="utf-8")
        cls.rebuilder_tree = ast.parse(cls.rebuilder_source)

    def test_slam_exposes_immutable_mapping_and_revision_contract(self):
        tracking = _class(self.roma_tree, "TrackingStatus")
        fields = {
            item.target.id
            for item in tracking.body
            if isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
        }
        self.assertTrue(
            {
                "pose_index",
                "keyframe_created",
                "keyframe_id",
                "geometry_revision",
                "geometry_corrected",
            }.issubset(fields)
        )
        slam = _class(self.roma_tree, "StereoSLAMv2")
        methods = {
            item.name for item in slam.body if isinstance(item, ast.FunctionDef)
        }
        self.assertIn("get_mapping_keyframe_snapshot", methods)
        self.assertIn("get_keyframe_pose_snapshot", methods)
        self.assertIn("get_pose_history_snapshot", methods)

    def test_committed_ba_and_loop_results_advance_geometry_revision(self):
        process = _method_source(
            self.roma_source, self.roma_tree, "StereoSLAMv2", "process_frame"
        )
        loop_apply = _method_source(
            self.roma_source,
            self.roma_tree,
            "StereoSLAMv2",
            "_apply_pending_loop_results",
        )
        backend_submit = _method_source(
            self.roma_source,
            self.roma_tree,
            "StereoSLAMv2",
            "_submit_keyframe_to_backend",
        )
        final_ba = _method_source(
            self.roma_source,
            self.roma_tree,
            "StereoSLAMv2",
            "final_optimization",
        )
        self.assertIn("self._geometry_corrected_in_process = False", process)
        self.assertIn("self._apply_pending_loop_results()", process)
        self.assertIn("self._mark_geometry_corrected()", backend_submit)
        self.assertIn("self._mark_geometry_corrected()", loop_apply)
        self.assertIn("self._mark_geometry_corrected()", final_ba)

    def test_mapper_stages_and_atomically_commits_map_transactions(self):
        self.assertNotIn("def _visible_points_and_labels", self.mapper_source)
        self.assertIn("KeyFrameObservationStore(", self.mapper_source)
        self.assertIn("memory_cache_size=", self.mapper_source)
        self.assertIn("compression_level=", self.mapper_source)
        rebuild = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_rebuild_historical_map",
        )
        self.assertIn("candidate, statistics", rebuild)
        self.assertIn("self.map_is_stale = True", rebuild)

        update = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_update_map",
        )
        self.assertIn("self._stage_keyframe_observation(new_observation)", update)
        self.assertIn("self._synchronize_map_state", update)
        self.assertNotIn("self.map_observations.add(new_observation)", update)

        incremental = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_commit_pending_map_observations",
        )
        self.assertIn("build_incremental_candidate", incremental)
        self.assertIn("self._commit_map_candidate", incremental)

        commit = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_commit_map_candidate",
        )
        self.assertLess(
            commit.index("self.map_observations.add_many"),
            commit.index("self.grid = candidate"),
        )
        self.assertIn("self.pending_map_observations.pop", commit)

    def test_failed_tracking_can_rebuild_history_without_fusing_frame(self):
        """失败帧先补交历史地图事务再退出；视觉融合只在守卫之后。"""
        callback = next(
            item
            for item in _class(
                self.mapper_tree, "RomaSemanticMapperNode"
            ).body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "_on_synchronized_input"
        )
        guard = next(
            node
            for node in ast.walk(callback)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Name)
            and node.test.operand.id == "vo_tracking_success"
        )
        synchronize_lines = [
            node.lineno
            for node in ast.walk(guard)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_synchronize_map_state"
        ]
        return_lines = [
            node.lineno
            for node in ast.walk(guard)
            if isinstance(node, ast.Return)
        ]
        update_lines = [
            node.lineno
            for node in ast.walk(callback)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_update_map"
        ]
        self.assertTrue(synchronize_lines)
        self.assertTrue(return_lines)
        self.assertTrue(update_lines)
        self.assertLess(min(synchronize_lines), max(return_lines))
        self.assertLess(max(return_lines), min(update_lines))

    def test_stale_and_pending_map_work_are_actively_retried(self):
        synchronize = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_synchronize_map_state",
        )
        self.assertIn("self.map_is_stale", synchronize)
        self.assertIn("self.pending_map_observations", synchronize)
        self.assertIn("self._rebuild_historical_map", synchronize)
        self.assertIn("self._commit_pending_map_observations", synchronize)

    def test_incomplete_evidence_blocks_publication_instead_of_being_skipped(self):
        synchronize = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_synchronize_map_state",
        )
        self.assertIn("Refusing to publish an incomplete map", synchronize)

        lidar_builder = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_make_keyframe_observation",
        )
        self.assertIn(
            "blocking map publication until a retry succeeds",
            lidar_builder,
        )
        self.assertNotIn("return None", lidar_builder)

        external_loss = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_update_map_without_vo_tracking",
        )
        self.assertIn(
            "self._record_missed_keyframe_observation",
            external_loss,
        )

        rebuild = _method_source(
            self.rebuilder_source,
            self.rebuilder_tree,
            "KeyFrameObservationStore",
            "rebuild",
        )
        self.assertIn("Cannot rebuild a complete map", rebuild)

    def test_snapshot_retries_pending_transaction_before_disk_write(self):
        snapshot = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "save_snapshot",
        )
        synchronize = snapshot.index("self._synchronize_map_state")
        disk_write = snapshot.index("self.grid.save_snapshot")
        self.assertLess(synchronize, disk_write)
        self.assertIn("publish=rclpy.ok(context=self.context)", snapshot)

    def test_ros_interfaces_and_numeric_config_are_unchanged(self):
        backup_interface = (
            BACKUP_ROOT
            / "t3_gridmap_v2_ws"
            / "src"
            / "t3_interfaces"
            / "msg"
            / "IncrementalSemanticMap.msg"
        )
        backup_config = (
            BACKUP_ROOT
            / "t3_gridmap_v2_ws"
            / "src"
            / "t3_semantic_mapping"
            / "config"
            / "semantic_map.yaml"
        )
        self.assertEqual(INTERFACE.read_bytes(), backup_interface.read_bytes())

        current_values = _config_parameter_values(CONFIG)
        backup_values = _config_parameter_values(backup_config)
        # 备案偏离一：机器相关路径改由 launch 提供，不再写入参数文件。
        for name in ("algorithm_root", "calib_path", "child_frame"):
            backup_values.pop(name)
            self.assertNotIn(name, current_values, name)
        # 备案偏离二：NaN 哨兵换成有限值（部分 ROS 2 发行版的
        # ParameterValue 拒绝 NaN）。
        for name in ("map_origin_x", "map_origin_y"):
            backup_values.pop(name)
            self.assertEqual(current_values.get(name), "-1.0e300", name)
        # 备案偏离三：enable_imu 是可切换的运行参数。将它
        # 留在 YAML 会在部分 ROS 2 发行版中反向覆盖 launch 参数，
        # 因此改由三个公共 launch 共享的定义提供。节点直接
        # 运行的默认值仍由 declare_parameter 保持为 true。
        backup_values.pop("enable_imu")
        self.assertNotIn("enable_imu", current_values)
        launch_options = LAUNCH_OPTIONS.read_text(encoding="utf-8")
        self.assertIn('"enable_imu"', launch_options)
        self.assertIn('default_value="true"', launch_options)
        # 备案偏离四：direct LIO 复用 keyframe_interval 作为独立的
        # 点云入图节拍。该值必须能被 launch 覆盖，节点与 launch 默认
        # 仍保持原来的 5，不改变既有默认行为。
        backup_values.pop("keyframe_interval")
        self.assertNotIn("keyframe_interval", current_values)
        self.assertIn('"keyframe_interval"', launch_options)
        self.assertIn('default_value="5"', launch_options)
        # 其余原始数值配置必须逐项保留；新增键允许（由参数契约测试
        # 的白名单另行约束）。
        for name, value in backup_values.items():
            self.assertEqual(current_values.get(name), value, name)

    def test_ros_parameter_names_and_defaults_are_unchanged(self):
        current = _declared_parameters(MAPPER)
        backup = _declared_parameters(BACKUP_MAPPER)
        # 新的全局闭环必须是显式选配，不得改变纯 VO 默认路径。
        self.assertIn(
            current.get("enable_global_loop_closure"),
            {"Constant(value=False)", "Constant(value=False, kind=None)"},
        )
        # 原始参数一个不能少。
        self.assertEqual(sorted(set(backup) - set(current)), [])
        # 备案偏离：NaN 默认值换成有限哨兵。
        for name in ("map_origin_x", "map_origin_y"):
            self.assertEqual(
                current.pop(name),
                "Name(id='_AUTO_MAP_ORIGIN', ctx=Load())",
            )
            backup.pop(name)
        # 其余原始参数默认值必须逐项一致。
        for name, default in backup.items():
            self.assertEqual(current.pop(name), default, name)
        # 剩下的只能是备案的 LiDAR/LIO/融合/瓦片参数族。
        unexpected = sorted(
            name
            for name in current
            if name not in _ALLOWED_NEW_PARAMETERS
            and not name.startswith(_ALLOWED_NEW_PARAMETER_PREFIXES)
        )
        self.assertEqual(unexpected, [])

    def test_ros_publishers_and_subscribers_are_unchanged(self):
        current = Counter(_ros_endpoints(MAPPER))
        backup = Counter(_ros_endpoints(BACKUP_MAPPER))
        # 原始发布/订阅端点（类型 + 消息 + 话题来源）必须全部保留。
        missing = list((backup - current).elements())
        self.assertEqual(missing, [])
        # 新增端点只允许是备案的 LiDAR/LIO 订阅，或不复制大数组的
        # full_tiles 批次元数据/同步回调完成回执发布者。旧地图载荷
        # 端点本身仍必须保留。
        for kind, message, topic in (current - backup).elements():
            if kind == "create_subscription":
                self.assertTrue(
                    any(
                        f"value='{name}'" in topic
                        for name in _ALLOWED_NEW_TOPIC_PARAMETERS
                    ),
                    topic,
                )
                continue
            self.assertEqual(kind, "create_publisher", topic)
            self.assertTrue(
                any(
                    message == expected_message
                    and f"value='{parameter}'" in topic
                    for expected_message, parameter in _ALLOWED_NEW_PUBLISHERS
                ),
                (message, topic),
            )

    def test_mapper_launches_share_one_mapping_parameter_definition(self):
        shared = LAUNCH_OPTIONS.read_text(encoding="utf-8")
        for parameter_name in (
            "incremental_publish_mode",
            "incremental_map_batch_topic",
            "incremental_map_replay_service",
            "publish_unchanged_map_outputs",
            "publish_layer_images",
            "publish_map_pointcloud",
            "path_publish_stride",
            "enable_lidar_mapping",
            "pointcloud_assist_pixel_radius",
            "map_admission_enabled",
            "map_observation_archive_enabled",
        ):
            self.assertIn(f'"{parameter_name}"', shared)
        for launch_path in MAPPER_LAUNCHES:
            source = launch_path.read_text(encoding="utf-8")
            self.assertIn("mapping_launch_arguments()", source)
            self.assertIn("**mapping_parameter_overrides()", source)
            self.assertNotIn(
                'DeclareLaunchArgument("enable_lidar_mapping"', source
            )

    def test_mapper_launches_share_one_input_transport_definition(self):
        shared = LAUNCH_OPTIONS.read_text(encoding="utf-8")
        for parameter_name in (
            "left_topic",
            "right_topic",
            "semantic_topic",
            "imu_topic",
            "stereo_input_reliability",
            "semantic_input_reliability",
            "imu_input_reliability",
            "image_input_qos_depth",
            "imu_input_qos_depth",
            "image_sync_mode",
            "image_sync_queue_size",
            "image_sync_slop_sec",
            "stereo_sync_max_skew_sec",
        ):
            self.assertIn(f'"{parameter_name}"', shared)
        for launch_path in MAPPER_LAUNCHES:
            source = launch_path.read_text(encoding="utf-8")
            self.assertIn("input_launch_arguments()", source)
            self.assertIn("**input_parameter_overrides()", source)

    def test_mapper_launches_share_one_odometry_output_definition(self):
        shared = LAUNCH_OPTIONS.read_text(encoding="utf-8")
        for parameter_name in (
            "odometry_camera_frame",
            "child_frame",
            "odometry_child_transform_confirmed",
            "odometry_child_to_camera_transform",
            "odometry_velocity_max_interval_sec",
            "odometry_unknown_covariance_variance",
        ):
            self.assertIn(f'"{parameter_name}"', shared)
        for launch_path in MAPPER_LAUNCHES:
            source = launch_path.read_text(encoding="utf-8")
            self.assertIn("localization_launch_arguments", source)
            self.assertIn("**localization_parameter_overrides()", source)

    def test_input_diagnostics_cannot_preempt_primary_synchronizer(self):
        interfaces = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_create_ros_interfaces",
        )
        primary = "self.synchronizer.registerCallback("
        diagnostic = "subscriber.registerCallback(self._record_image_input"
        self.assertLess(interfaces.index(primary), interfaces.index(diagnostic))

        recorder = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_record_image_input",
        )
        self.assertIn("except (AttributeError, TypeError, ValueError)", recorder)
        self.assertNotIn("stamp_nanoseconds(", recorder)

    def test_imu_and_map_publication_optimization_is_lossless(self):
        interfaces = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_create_ros_interfaces",
        )
        self.assertIn("self.imu_callback_group", interfaces)
        self.assertIn("callback_group=self.imu_callback_group", interfaces)

        main_node = next(
            item
            for item in self.mapper_tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "main"
        )
        main_source = ast.get_source_segment(
            self.mapper_source, main_node
        ) or ""
        self.assertIn("int(node.enable_imu)", main_source)
        self.assertIn("MultiThreadedExecutor", main_source)
        self.assertIn("num_threads=executor_threads", main_source)

        publication = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_publish_map_if_changed",
        )
        self.assertIn("publish_unchanged_map_outputs", publication)
        self.assertLess(
            publication.index("self._publish_current_map_outputs"),
            publication.index("self._last_published_map_signature ="),
        )

        signature = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_map_output_signature",
        )
        self.assertIn("int(self.map_revision)", signature)
        self.assertIn("self.grid.window_geometry", signature)

        outputs = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_publish_map_outputs",
        )
        for required_call in (
            "self.occupancy_pub.publish",
            "self.grid_map_pub.publish",
            "self._publish_incremental_",
        ):
            self.assertIn(required_call, outputs)
        layer_branch = outputs[
            outputs.index("if self.publish_layer_images"):
            outputs.index("layer_map =")
        ]
        for formal_output in (
            "occupancy_pub",
            "grid_map_pub",
            "incremental_map_pub",
        ):
            self.assertNotIn(formal_output, layer_branch)

        synchronize = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_synchronize_map_state",
        )
        self.assertIn("self._publish_map_if_changed", synchronize)
        self.assertNotIn("self._publish_current_map_outputs", synchronize)

        pose = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_publish_pose",
        )
        self.assertIn("self.odometry_pub.publish(odometry)", pose)
        self.assertIn("self._publish_path_if_due()", pose)

        path_policy = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "_publish_path_if_due",
        )
        self.assertIn("self.path_publish_stride", path_policy)

    def test_shutdown_commits_backend_before_rebuild_and_snapshot(self):
        close = _method_source(
            self.mapper_source,
            self.mapper_tree,
            "RomaSemanticMapperNode",
            "close_resources",
        )
        shutdown = close.index("shutdown()")
        rebuild = close.index("self._synchronize_map_state")
        snapshot = close.index("self.save_snapshot()")
        self.assertLess(shutdown, rebuild)
        self.assertLess(rebuild, snapshot)
        self.assertIn("can_publish = rclpy.ok", close)
        self.assertIn("publish=can_publish", close)


if __name__ == "__main__":
    unittest.main()
