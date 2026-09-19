# 该文件是预处理流水线的总调度文件，把一条 Aria 录像（sample.vrs + MPS）按固定顺序跑完，最后写出训练用的 training_data.json
# -*- coding: utf-8 -*-  编码声明：显式声明该 Python 文件使用 UTF-8 字符编码 保存和读取，一个防错规范，如果是 python3 则不需要该编码声明
# @FileName: Preprocess.py  文件名标注，给人或工具看，不影响运行。

# ”“”...“”“ 是多行注释，给人或工具看，不影响运行，放在文件前面时，会成为模块的__doc__，用来写用途、流水线步骤、技术要点。运行时不会当代码执行。

"""
====================================================================================================
Preprocessing Orchestrator (Preprocess.py)
====================================================================================================

Description:
    This script serves as the central orchestrator for the entire Project Aria data preprocessing 
    pipeline. It sequentially manages the extraction of raw sensor data (AriaCam 相机, AriaHands 手, 
    AriaSlam SLAM), temporal segmentation (AriaPhases), visual semantic processing (DINOSAM, Lama), 
    3D kinematics reconstruction 三维运动学重建 (CoTracker, CamTriangulator), and dataset generation 数据集生成 (DatasetGen).

Core Functionalities:
    1.  Initialization: Sets up VRS and MPS data providers and establishes global frame bounds.
        初始化：设置 VRS 和 MPS 数据提供者，并建立全局帧边界。
    2.  Base Extraction (Aria): Processes camera streams, hand tracking, and SLAM poses .
        基础提取 (Aria)：处理相机流、手跟踪和 SLAM 位姿。
    3.  Phase-based Indexing: Segments the video sequence into logical states (Navigation 导航, 
        Transition 过渡, Manipulation 操作) and isolates an 'Object-Centric' temporal window.
        基于阶段的分段索引：将视频序列分为逻辑状态（导航、过渡、操作），并隔离一个“对象中心”的时间窗口。
    4.  Visual-Spatial Pipeline: Executes deep learning models (DINO, SAM2, CoTracker, LaMa) 
        and geometric triangulators to establish multi-object 6-DOF tracking.
        视觉空间管道：执行深度学习模型（DINO, SAM2, CoTracker, LaMa）和几何三角测量器，建立多对象 6-DOF 跟踪。
    5.  Consolidation: Generates the final training_data.json and comprehensive 
        video/3D visualizations.
        整合：生成最终的 training_data.json 和全面的视频/3D 可视化。

Technical Specifics:
    - Recursive Configuration: Loads a master YAML that inherently points to sub-module configs.
        递归配置：加载主 YAML，自动解析所有嵌套的子模块配置。
    - Decorators: Uses @time_it for precise performance profiling of each pipeline stage.
        装饰器：使用 @time_it 进行每个流水线阶段的精确性能分析。
====================================================================================================
"""
import os
import argparse
import shutil
import time
from tqdm import tqdm
import json  # 读写 json
import re
import traceback
import cv2
# 导入，项目内部模块

# Project Aria SDK（Software Development Kit）
from projectaria_tools.core import data_provider, mps  # 打开 .vrs录像，读相机流、时间戳
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.sensor_data import TimeDomain

from preprocess.AriaCam import AriaCamGenerator
from preprocess.AriaHands import AriaHandsGenerator
from preprocess.AriaSlam import AriaSlamGenerator
from preprocess.AriaPhases import AriaPhasesGenerator
from preprocess.DINOSAMOps import run_dinosam, print_dinosam_stats
from preprocess.KptsSelector import run_kptsselector
from preprocess.CoTrackerOffline import run_cotracker_offline, print_cotracker_offline_stats
from preprocess.CamTriangulator import run_camtriagulator
from preprocess.Lama import run_lama, print_lama_stats
from preprocess.VisualKpts import run_visualkpts
from preprocess.DatasetGen import run_datasetgen
from preprocess.HandTrackingComparison import run_hand_tracking_comparison, HAND_ENTITY_KEYS


from utils.utils_media import create_video_from_frames
from utils.utils_math import time_it
from utils.utils_io import load_cfg_dynamic_task

# Hand detection method registry: method_name → (module_path, entry_function_name)
HAND_METHOD_REGISTRY = {
    "mediapipe": ("preprocess.MediaPipeHands", "run_mediapipe_hands"),
    "wilor":     ("preprocess.WiLoRHands",     "run_wilor_hands"),
    "hamer":     ("preprocess.HaMeRHands",     "run_hamer_hands"),
}


def get_task_list(parent_path, run_range=None):
    """
    Scans the parent directory for folders matching the 'mps_xxxx_x_xxx_vrs' pattern.
    Extracts the numeric index, sorts them numerically, and filters by the provided range.
    """
    # Regex pattern: 'mps_' followed by any characters, an underscore, numeric digits, and '_vrs'
    pattern = re.compile(r'^mps_.*_(\d+)_vrs$')
    
    task_folders = []
    if not os.path.exists(parent_path):
        print(f"║ [Error] Parent path not found: {parent_path}")
        return []

    # Iterate through directory items
    for item in os.listdir(parent_path):
        full_path = os.path.join(parent_path, item)
        if os.path.isdir(full_path):
            match = pattern.match(item)
            if match:
                # Extract digits and convert to int for proper numerical sorting
                index = int(match.group(1)) 
                task_folders.append({
                    "index": index,
                    "path": full_path
                })
    
    # Sort folders by the numeric index (ascending)
    task_folders.sort(key=lambda x: x["index"])
    
    # Apply range filtering [start, end] inclusive
    if run_range:
        start, end = run_range
        task_folders = [f for f in task_folders if start <= f["index"] <= end]
        print(f"║ [Range] Filtering indices from {start} to {end}. Found {len(task_folders)} task(s).")
    
    return [f["path"] for f in task_folders]


class Preprocess:
    """
    The master orchestrator class executing the end-to-end preprocessing pipeline.
    """
    def __init__(self, mps_path, cfg_path, task, export_video, export_gif):

        self.mps_path = mps_path
        self.cfg_path = cfg_path
        self.task = task
        self.export_video = export_video
        self.export_gif = export_gif
        
        # Recursively loads the master config and automatically resolves all nested sub-configs
        self.cfg = load_cfg_dynamic_task(self.cfg_path, self.mps_path, self.task)


    @time_it
    def init_preprocess(self) -> None:
        """
        Initializes the Project Aria VRS/MPS data providers and determines the valid 
        frame range based on the intersection of RGB timestamps and Hand Tracking data.
        """

        # Providers
        self.vrs_provider = data_provider.create_vrs_data_provider(os.path.join(self.mps_path, "sample.vrs"))
        self.mps_provider = MpsDataProvider(MpsDataPathsProvider(self.mps_path).get_data_paths())

        # Range
        self.aria_hands_mps = mps.hand_tracking.read_hand_tracking_results(os.path.join(self.mps_path, "hand_tracking", "hand_tracking_results.csv"))
        rgb_tss = self.vrs_provider.get_timestamps_ns(self.vrs_provider.get_stream_id_from_label("camera-rgb"), TimeDomain.DEVICE_TIME)
        self.num_total_frames = len(self.aria_hands_mps)
        self.start_idx = len(rgb_tss) - len(self.aria_hands_mps)
        self.end_idx = self.start_idx + len(self.aria_hands_mps) - 1
        print(f"║ [System] Initialization Complete. start_idx: {self.start_idx}, end_idx: {self.end_idx}")


    @time_it
    def preprocess_aria(self) -> None:
        """
        Extracts foundational device data: Camera frames (rectified), Hand Tracking, 
        SLAM kinematics, and Temporal Task Phases. Renders the baseline Aria visualization.
        """

        # AriaCam
        aria_cam_rgb_generator = AriaCamGenerator(self.mps_path, self.cfg.AriaCam_path, self.vrs_provider, self.mps_provider, label='rgb')
        aria_cam_rgb = aria_cam_rgb_generator.get_aria_cam(self.start_idx, self.end_idx)
        tss = aria_cam_rgb.tss
        aria_cam_rgb.save_aria_cam_json(label='rgb')
        aria_cam_rgb.save_aria_cam_video_orig(self.export_video, self.export_gif, label='rgb')

        # AriaHands
        aria_hands_generator = AriaHandsGenerator(self.mps_path, self.cfg.AriaHands_path, self.aria_hands_mps, aria_cam_rgb)
        aria_hands = aria_hands_generator.get_aria_hands()
        aria_hands.save_aria_hands_json()

        # AriaSlam
        aria_slam_generator = AriaSlamGenerator(self.mps_path, self.cfg.AriaSlam_path, aria_cam_rgb)
        aria_slam = aria_slam_generator.get_aria_slam()
        aria_slam.save_aria_slam_json()

        # AriaPhases
        aria_phases_generator = AriaPhasesGenerator(self.mps_path, self.cfg.AriaPhases_path, aria_cam_rgb, aria_slam, aria_hands)
        aria_phases = aria_phases_generator.get_aria_phases()
        aria_phases.save_aria_phases_json()
        
        # vis — produces TWO videos per frame in a single pass:
        #   aria_vis.mp4             — simplified gripper view (wrist + thumb + index)
        #   aria_vis_full_hands.mp4  — full 21-keypoint hand skeleton (all fingers + bones)
        # Both share the same SLAM/Phases/HUD overlays; only the hand-skeleton layer differs.
        if self.export_video:
            frames_simplified = []
            frames_full = []
            for idx, ts in enumerate(tqdm(tss, desc="Aria Visualization")):
                img_base = aria_cam_rgb.cam[idx].img.copy()

                for full_skeleton, frames_list in [(False, frames_simplified),
                                                   (True,  frames_full)]:
                    img = img_base.copy()

                    # Overlay: Hands Skeleton (simplified gripper or full 21-keypoint)
                    img = aria_hands_generator.draw_aria_hands_skeleton(
                        img, aria_hands.hands[idx],
                        aria_cam_rgb.cam[idx].k, aria_cam_rgb.cam[idx].d, aria_cam_rgb.cam[idx].c2w,
                        full_skeleton=full_skeleton,
                    )

                    # Overlay: Hands HUD Panel
                    img = aria_hands_generator.draw_aria_hands_panel(img, idx, aria_hands.hands[idx])

                    # Overlay: Slam 3D Trajectory Projection
                    img = aria_slam_generator.draw_future_traj_on_image(img, idx, aria_slam)

                    # Overlay: Slam HUD Panel
                    img = aria_slam_generator.draw_aria_slam_panel(img, aria_slam.frames[idx], aria_slam)

                    # Overlay: Phases HUD Panel
                    img = aria_phases_generator.draw_aria_phases_panel(img, idx, aria_phases)

                    frames_list.append(img)

            create_video_from_frames(
                frames=frames_simplified,
                save_path=os.path.join(self.mps_path, "preprocess", "vis", "aria_vis.mp4"),
                fps=self.cfg.AriaCam.fps,
                export_gif=self.export_gif
            )
            create_video_from_frames(
                frames=frames_full,
                save_path=os.path.join(self.mps_path, "preprocess", "vis", "aria_vis_full_hands.mp4"),
                fps=self.cfg.AriaCam.fps,
                export_gif=self.export_gif
            )
            

    @time_it
    def preprocess_indices(self) -> None:
        """
        Parses the computed phase states to categorize frames into Navigation, Transition, 
        and Manipulation segments. Extracts the crucial 'Object-Centric' temporal window.
        """

        phases_json_path = os.path.join(self.mps_path, "preprocess", "aria_phases_results.json")
        with open(phases_json_path, 'r') as f: 
            phases_data = json.load(f)
        
        windows_dict = phases_data.get("stage_window_check", {}).get("windows", {})
        
        def get_image_list_from_keys(keys):
            allowed_idx = []
            for key in keys:
                wins = windows_dict.get(key, [])
                for win in wins:
                    if not isinstance(win, (list, tuple)) or len(win) != 2:
                        continue
                    s, e = int(win[0]), int(win[1])
                    r_start, r_end = (s, e) if s <= e else (e, s)
                    allowed_idx.extend(range(r_start, r_end + 1))
            allowed_idx = sorted(set(allowed_idx))
            res_paths = []
            for i in allowed_idx:
                img_path = os.path.join(self.mps_path, "preprocess", "all_data", f"{i:05d}", "rgb.png")
                if os.path.exists(img_path):
                    res_paths.append(img_path)
            return res_paths
        
        self.manip_image_list = get_image_list_from_keys(["0", "4"])
        self.nav_image_list = get_image_list_from_keys(["1", "2"])
        self.transition_image_list = get_image_list_from_keys(["3"])

        self.raw_manip_image_list = get_image_list_from_keys(["0", "3", "4"])

        self.all_image_list = [] # actually it's equal to manip_image_list+nav_image_list+transition_image_list
        for i in range(0, self.num_total_frames):
            p = os.path.join(self.mps_path, "preprocess", "all_data", f"{i:05d}", "rgb.png")
            if os.path.exists(p):
                self.all_image_list.append(p)
        
        if not self.manip_image_list:
            print("║ [Warning] Manip (Phase 0) images not found. Manip training data will be empty.")
        if not self.nav_image_list:
            print("║ [Warning] Nav (Phase 1/2) images not found. Nav training data will be empty.")
        
        print(f"║ [Info] Indices Split: Manip={len(self.manip_image_list)} frames, Nav={len(self.nav_image_list)} frame, Transition={len(self.transition_image_list)} frames, Full Sequence: {len(self.all_image_list)} frames, raw_manip_image_list: {len(self.raw_manip_image_list)} frames")
        
        # --- Resolve Object-Centric Image List ---
        if self.raw_manip_image_list:
            first_manip = self.raw_manip_image_list[0]
            try:
                split_idx = self.all_image_list.index(first_manip)
            except ValueError:
                split_idx = 0
        pre_frames_count = min(split_idx, self.cfg.CoTracker.object_centric_max_frames)
        self.object_centric_image_list = self.all_image_list[split_idx - pre_frames_count : split_idx]
        if len(self.object_centric_image_list) < self.cfg.CoTracker.object_centric_min_frames:
            needed = self.cfg.CoTracker.object_centric_min_frames - len(self.object_centric_image_list)
            supplementary_frames = self.raw_manip_image_list[:needed]
            self.object_centric_image_list = self.object_centric_image_list + supplementary_frames
            print(f"║ [Info] Pre-manip frames only {pre_frames_count}, supplemented {len(supplementary_frames)} frames from manip_list to reach 30.")

        actual_ref_idx = self.cfg.CoTracker.ref_idx if abs(self.cfg.CoTracker.ref_idx) < len(self.object_centric_image_list) else -1
        self.object_ref_img_path = self.object_centric_image_list[actual_ref_idx]

        print(f"║ [Info] Object-Centric sequence length: {len(self.object_centric_image_list)}")
        print(f"║ [Info] Selecting reference frame: {self.object_ref_img_path}")


    @time_it
    def preprocess_dinosam(self) -> None:
        """
        Executes Grounding DINO and SAM2 for bounding box detection and mask segmentation.
        """
        
        frames_all_dinosam = []
        dinosam_image_list = self.object_centric_image_list + self.raw_manip_image_list
        print(f"║ [Info] DINOSAM/VisualGuide Processing {len(dinosam_image_list)} frames...")
        for i, img_path in enumerate(tqdm(dinosam_image_list, desc="DINO+SAM2+Guide Pipeline")):
            if not os.path.exists(img_path): continue
            vis_dinosam = run_dinosam(
                cfg_path=self.cfg.DINOSAM_path,
                image_path=img_path,
            )

            if self.export_video:
                frames_all_dinosam.append(vis_dinosam)

        print_dinosam_stats()

        # if self.export_video:
        #     create_video_from_frames(
        #         frames=frames_all_dinosam,
        #         save_path=os.path.join(self.mps_path, "preprocess", "vis", "mask_vis.mp4"),
        #         fps=self.cfg.AriaCam.fps,
        #         export_gif=self.export_gif
        #     )
        
    
    @time_it
    def preprocess_kptsselector(self) -> None:
        """
        Extracts equidistant tracking keypoints from the segmented masks of the reference frame.
        """
        
        ref_frame_dir = os.path.dirname(self.object_ref_img_path)
        obj_keys =[k for k in self.cfg.DINOSAM.dinosam_prompt.keys() if k.startswith("obj")]
        
        all_kpts_data = {}
        
        for obj_key in obj_keys:
            src_mask_path = os.path.join(ref_frame_dir, f"mask_{obj_key}.png")
            dst_mask_path = os.path.join(self.mps_path, "preprocess", f"dinosam_mask_{obj_key}.png")
            
            if os.path.exists(src_mask_path):
                shutil.copy2(src_mask_path, dst_mask_path)
                print(f"║ [Info] Successfully copied {src_mask_path} to {dst_mask_path}")
                
                kpts = run_kptsselector(
                    cfg_path=self.cfg.KptsSelector_path,
                    mask_path=dst_mask_path,
                    save_img_path=os.path.join(self.mps_path, "preprocess", f"kptsselector_vis_{obj_key}.png"),
                    rgb_path=self.object_ref_img_path,
                )
                
                if kpts:
                    all_kpts_data[obj_key] = kpts
            else:
                print(f"║ [Warning] Expected mask file NOT found: {src_mask_path}")

        save_json_path = os.path.join(self.mps_path, "preprocess", "kptsselector_results.json")
        data = {
            "method": "AUTO_MASK_PCA_EDGE",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "objects": all_kpts_data  # {"obj_1": [[u,v]...], "obj_2": [[u,v]...]}
        }
        with open(save_json_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"║ [Output] Multi-object Keypoints JSON saved: {save_json_path}")


    @time_it
    def preprocess_cotracker(self) -> None:
        """
        Executes CoTracker3 for dense point tracking across the active sequence.
        """

        frames_all_cotracker = []
        cotracker_image_list = self.object_centric_image_list + self.raw_manip_image_list
        print(f"║ [Info] CoTracker Processing {len(cotracker_image_list)} frames...")
        for i, img_path in enumerate(tqdm(cotracker_image_list, desc="CoTracker Pipeline")):
            if not os.path.exists(img_path): continue
            vis_cotracker = run_cotracker_offline(
                image_path=img_path,
                cfg_path=self.cfg.CoTracker_path,
                frame_idx=i,
                all_image_paths=cotracker_image_list,
                mps_path=self.mps_path,
            )


            if self.export_video:
                frames_all_cotracker.append(vis_cotracker)

        print_cotracker_offline_stats()

        # if self.export_video:
        #     create_video_from_frames(
        #         frames=frames_all_cotracker,
        #         save_path=os.path.join(self.mps_path, "preprocess", "vis", "cotracker_vis.mp4"),
        #         fps=self.cfg.AriaCam.fps,
        #         export_gif=self.export_gif
        #     )



    @time_it
    def preprocess_camtriangulator(self) -> None:
        """
        Triangulates 2D tracks into 3D points and estimates 6-DOF object poses.
        """
        
        frames_all_camtriagulator = []
        camtriangulator_image_list = self.object_centric_image_list
        print(f"║ [Info] CamTriangulator Processing {len(camtriangulator_image_list)} frames...")
        for i, img_path in enumerate(tqdm(camtriangulator_image_list, desc="CamTriangulator Pipeline")):
            if not os.path.exists(img_path): continue

            vis_camtriagulator = run_camtriagulator(
                image_path=img_path,
                cfg_path=self.cfg.CamTriangulator_path,
                frame_idx=i,
                all_image_paths=camtriangulator_image_list,
                mps_path=self.mps_path,
            )

            if self.export_video:
                frames_all_camtriagulator.append(vis_camtriagulator)

        # if self.export_video:
        #     create_video_from_frames(
        #             frames=frames_all_camtriagulator,
        #             save_path=os.path.join(self.mps_path, "preprocess", "vis","camtriagulator_vis.mp4"),
        #             fps=self.cfg.AriaCam.fps,
        #             export_gif=self.export_gif
        #     )


    @time_it
    def preprocess_lama(self) -> None:
        """
        Uses LaMa to perform background inpainting (removing the arm and objects).
        """
        
        frames_all_lama = []
        lama_image_list = self.object_centric_image_list + self.raw_manip_image_list
        print(f"║ [Info] Lama Processing {len(lama_image_list)} frames...")
        for i, img_path in enumerate(tqdm(lama_image_list, desc="Lama Pipeline")):
            if not os.path.exists(img_path): continue
            vis_lama = run_lama(
                image_path=img_path,
                cfg_path=self.cfg.Lama_path
            )

            if self.export_video:
                frames_all_lama.append(vis_lama)
        
        print_lama_stats()

        # if self.export_video:
        #     create_video_from_frames(
        #         frames=frames_all_lama,
        #         save_path=os.path.join(self.mps_path, "preprocess", "vis", "lama_vis.mp4"),
        #         fps=self.cfg.AriaCam.fps,
        #         export_gif=self.export_gif
        #     )


    @time_it
    def preprocess_visualkpts(self) -> None:
        """
        Renders the aesthetic visual keypoints (gripper wireframes and multi-object trails).
        """
        
        frames_all_visualkpts = []
        visualkpts_image_list = self.object_centric_image_list + self.raw_manip_image_list
        print(f"║ [Info] VisualKpts Processing {len(visualkpts_image_list)} frames...")
        for i, img_path in enumerate(tqdm(visualkpts_image_list, desc="VisualKpts Pipeline")):
            if not os.path.exists(img_path): continue
            vis_visualkpts = run_visualkpts(
                    image_path=img_path,
                    cfg_path=self.cfg.VisualKpts_path,
                    frame_idx=i,
                    all_image_paths=visualkpts_image_list,
                    mps_path=self.mps_path,
                )

            if self.export_video:
                frames_all_visualkpts.append(vis_visualkpts)

        if self.export_video:
            create_video_from_frames(
                frames=frames_all_visualkpts,
                save_path=os.path.join(self.mps_path, "preprocess", "vis", "visualkpts_vis.mp4"),
                fps=self.cfg.AriaCam.fps,
                export_gif=self.export_gif
            )


    @time_it
    def preprocess_hands(self) -> None:
        """
        Runs additional hand detection methods (MediaPipe, WiLoR, HaMeR) as specified
        in the config's `hand_tracking_methods` list. Each method produces per-frame
        JSON files (e.g., mediapipe_hands.json) in the same format as aria_hands.json.

        Config example:
            hand_tracking_methods: ["aria_mps", "mediapipe", "wilor", "hamer"]

        'aria_mps' is always handled by preprocess_aria() and is skipped here.
        Methods not listed are simply not run. Missing methods are handled gracefully
        downstream (DatasetGen writes null for ungenerated methods).
        """
        methods = getattr(self.cfg, 'hand_tracking_methods', None)
        if not methods:
            print("║ [Info] No hand_tracking_methods configured. Skipping alternative hand detection.")
            return

        # Filter to only non-aria methods that need to be run
        methods_to_run = [m for m in methods if m != "aria_mps" and m in HAND_METHOD_REGISTRY]
        if not methods_to_run:
            print("║ [Info] No additional hand methods to run (only aria_mps configured).")
            return

        print(f"║ [Info] Running alternative hand detection methods: {methods_to_run}")

        for method_name in methods_to_run:
            module_path, func_name = HAND_METHOD_REGISTRY[method_name]
            print(f"\n║ ┌── Running hand detection: {method_name}")
            try:
                import importlib
                mod = importlib.import_module(module_path)
                run_func = getattr(mod, func_name)

                # Build the config path for this method (optional, may not exist)
                method_cfg_path = getattr(
                    self.cfg,
                    f'{method_name.capitalize()}Hands_path',
                    self.cfg.AriaHands_path   # fallback to AriaHands config
                )

                run_func(
                    mps_path=self.mps_path,
                    cfg_path=method_cfg_path,
                    export_video=self.export_video,
                    export_gif=self.export_gif,
                )
                print(f"║ └── {method_name}: Done ✓")

            except ImportError as e:
                print(f"║ └── {method_name}: SKIPPED (import error: {e})")
                print(f"║     Make sure the package is installed in the current environment.")
            except Exception as e:
                print(f"║ └── {method_name}: FAILED ({e})")
                traceback.print_exc()


    @time_it
    def preprocess_datasetgen(self) -> None:
        """
        Consolidates spatial kinematics into the final training_data.json.
        """

        datasetgen_image_list = self.manip_image_list
        if datasetgen_image_list:
            print(f"║ [Info] Generating training_data.json for training...")
            ds_stats = run_datasetgen(
                image_list=datasetgen_image_list,
                mps_path=self.mps_path,
                cfg_path=self.cfg.DatasetGen_path,
            )
        else:
            print("║ [Skip] datasetgen image_list is empty. Skipping dataset generation.")


    @time_it
    def preprocess_hand_tracking_comparison(self) -> None:
        """
        Generates per-method hand tracking visualizations and metrics.
        Output: {mps_path}/preprocess/hand_tracking/{method}/... + comparison plots.
        """
        try:

            methods = getattr(self.cfg, "hand_tracking_methods",
                              ["aria_mps", "mediapipe", "wilor", "hamer"])

            # Only include methods that were actually run (have JSON files)
            all_data_dir = os.path.join(self.mps_path, "preprocess", "all_data")
            first_frame = None
            for d in sorted(os.listdir(all_data_dir)):
                if os.path.isfile(os.path.join(all_data_dir, d, "training_data.json")):
                    first_frame = os.path.join(all_data_dir, d, "training_data.json")
                    break
            if first_frame is None:
                print("║ [Skip] No training_data.json found. Skipping hand tracking comparison.")
                return

            with open(first_frame) as f:
                td = json.load(f)

            available_methods = []
            for m in methods:
                key = HAND_ENTITY_KEYS.get(m, "hands")
                if key in td.get("entities", {}) and td["entities"][key] is not None:
                    available_methods.append(m)

            if len(available_methods) < 1:
                print("║ [Skip] No hand tracking methods have data. Skipping comparison.")
                return

            print(f"║ [HandTrackingComparison] Running for: {', '.join(available_methods)}")
            run_hand_tracking_comparison(
                mps_path=self.mps_path,
                methods=available_methods,
                side=getattr(self.cfg, "single_hand_side", "right"),
            )
        except ImportError as e:
            print(f"║ [Skip] HandTrackingComparison not available: {e}")
        except Exception as e:
            print(f"║ [WARN] Hand tracking comparison failed: {e}")
            traceback.print_exc()


    @time_it
    def run(self) -> None:
        """
        Executes the entire preprocessing workflow in sequence.
        """

        self.init_preprocess()
        self.preprocess_aria()
        self.preprocess_indices()                                          
        self.preprocess_dinosam()
        self.preprocess_kptsselector()
        self.preprocess_cotracker()
        self.preprocess_camtriangulator()
        self.preprocess_lama()
        self.preprocess_visualkpts()
        # self.preprocess_hands()
        self.preprocess_datasetgen()
        # self.preprocess_hand_tracking_comparison()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   # mps_path can now be a single task folder OR a parent directory containing multiple tasks
    parser.add_argument("--mps_path", type=str, required=True, 
                        help="Path to a single MPS directory or a parent directory for batch processing")
    
    parser.add_argument("--cfg_path", type=str, default="./cfg/preprocess/base/Preprocess.yaml", 
                        help="Path to the base Preprocess.yaml")
    
    parser.add_argument("--task", type=str, default=None, 
                        help="Task name (e.g., serve_bread) to load specific YAML overrides")
    
    parser.add_argument("--range", type=int, nargs=2, metavar=('START', 'END'), 
                        help="Numeric range of indices to process (e.g., --range 0 10)")
    
    parser.add_argument("--no-video", action="store_false", dest="export_video", help="Disable MP4 video export")
    parser.add_argument("--no-gif", action="store_false", dest="export_gif", help="Disable GIF animation export")
    args = parser.parse_args()

    # 1. Identify Task Execution Mode
    if os.path.exists(os.path.join(args.mps_path, "sample.vrs")):
        final_tasks = [args.mps_path]
        print(f"║ [Mode] Single task detected: {os.path.basename(args.mps_path)}")
    else:
        # Otherwise, scan for sub-folders matching the naming convention
        print(f"║ [Mode] Batch directory detected. Scanning for sub-tasks...")
        final_tasks = get_task_list(args.mps_path, args.range)

    if not final_tasks:
        print("║ [Error] No valid MPS task folders found. Terminating execution.")
        exit(1)

    # 2. Sequential Pipeline Execution
    print(f"║ [Batch] Starting sequence for {len(final_tasks)} task(s)...")
    
    for i, path in enumerate(final_tasks):
        folder_name = os.path.basename(path)
        print(f"\n{'='*100}")
        print(f"║ [{i+1}/{len(final_tasks)}] Processing: {folder_name}")
        print(f"{'='*100}\n")

        # Reset all singletons between sessions to avoid stale state
        from preprocess.CoTrackerOffline import reset_cotracker_offline
        from preprocess.VisualKpts import reset_visualkpts
        reset_cotracker_offline()
        reset_visualkpts()

        try:
            preprocess_engine = Preprocess(
                mps_path=path,
                cfg_path=args.cfg_path,
                task=args.task,
                export_video=args.export_video,
                export_gif=args.export_gif
            )
            preprocess_engine.run()

        except Exception as e:
            print(f"\n║ [Critical Error] Task failed: {folder_name}")
            print(traceback.format_exc())
            print(f"║ [System] Skipping to next task...\n")
            continue 

    print(f"\n║ [Done] All tasks in the batch have been processed.")


# python -m preprocess.Preprocess --mps_path  "./data/serve_bread/aria/mps_serve_bread_050_vrs" --task serve_bread
# python -m preprocess.Preprocess --mps_path "./data/serve_bread/aria/" --task serve_bread --range 40 61