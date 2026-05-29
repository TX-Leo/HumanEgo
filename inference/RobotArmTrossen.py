# -*- coding: utf-8 -*-
import sys
import time
import argparse
import threading
import numpy as np
from typing import List, Union, Optional, Dict, Tuple
from scipy.spatial.transform import Rotation as R
import cv2
import trossen_arm

from utils.utils_io import load_cfg


class RobotArmTrossen():
    def __init__(self, cfg_path: str):
        self.cfg_path = cfg_path          # keep for saving back
        self.cfg = load_cfg(cfg_path)
        self._fix_rotation = False
        self._fixed_rot_raw = None  # driver-native rotvec, captured once
        self.init()
        

    def init(self) -> bool:
        self.driver = trossen_arm.TrossenArmDriver()
        self._is_draggable = False
        try:
            print(f"║ [RobotArmTrossen] Connecting to {self.cfg.ip}...")
            self.init_arm(clear_error_on_init=True)
            self.init_T_base_in_cam()
            self.init_gripper()
            self.print_all_info()
            # self.go_initial()
            return True
        except Exception as e:
            print(f"║ [RobotArmTrossen] Init Failed: {e}")
            return False
        


    def init_arm(self, clear_error_on_init=False):
        model = getattr(trossen_arm.Model, self.cfg.model_type)
        ee_type = getattr(trossen_arm.StandardEndEffector, self.cfg.ee_type)
        self.driver.configure(
            model,
            ee_type,
            self.cfg.ip,
            clear_error_on_init
        )
        self.driver.set_all_modes(trossen_arm.Mode.position)


    def init_T_base_in_cam(self):
        if "T_base_in_cam" in self.cfg:
            self._T_base_in_cam = np.array(self.cfg.T_base_in_cam)
        else:
            self._T_base_in_cam = np.eye(4)

    def save_T_base_in_cam(self):
        """Write current _T_base_in_cam back to the arm's YAML config file."""
        import yaml, re
        with open(self.cfg_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Build a nicely-formatted replacement string
        T = self._T_base_in_cam
        rows = []
        for i in range(4):
            elems = ', '.join(f'{T[i, j]:<16.8g}' for j in range(4))
            rows.append(f'    [{elems}]')
        replacement = 'T_base_in_cam: [\n' + ',\n'.join(rows) + '\n]'

        # Replace the ACTIVE (non-commented) T_base_in_cam block
        # Match from "T_base_in_cam:" at line start to the closing "]" on its own line
        pattern = r'(?m)^T_base_in_cam:\s*\[.*?\n\]'
        new_text = re.sub(pattern, replacement, text, count=1, flags=re.DOTALL)
        with open(self.cfg_path, 'w', encoding='utf-8') as f:
            f.write(new_text)
        print(f"║ [RobotArm] T_base_in_cam saved → {self.cfg_path}")


    def init_gripper(self):
        self.ee = self.driver.get_end_effector()
        self._gripper_max = abs(self.ee.offset_finger_left) + abs(self.ee.offset_finger_right)
        print(f'orig self.ee.t_flange_tool: {self.ee.t_flange_tool}')
        print(f'orig _gripper_max: {self._gripper_max}')
        self.ee.t_flange_tool = list(self.cfg.t_flange_tool)
        self.driver.set_end_effector(self.ee)
        self._gripper_max = 0.045
        print(f'New self.ee.t_flange_tool: {self.ee.t_flange_tool}')
        print(f'New _gripper_max: {self._gripper_max}')
        
        try:
            all_limits = self.driver.get_joint_limits()
            gripper_limit = all_limits[-1]
            print(f"Original Gripper Limit: min={gripper_limit.position_min}, max={gripper_limit.position_max}")
            gripper_limit.position_max = 0.044999999999
            gripper_limit.position_min = 0.00
            self.driver.set_joint_limits(all_limits)
            print(f"New Gripper Limit: min={gripper_limit.position_min}, max={gripper_limit.position_max}")
            print("Successfully updated gripper joint limits.")
        except Exception as e:
            print(f"Failed to update joint limits: {e}")
    

    def recover_connection(self):
        self.driver.cleanup()
        self.init_arm(clear_error_on_init=True)
        
    
    def set_mode(self, mode: str) -> None:
        if mode == "position":
            self.driver.set_all_modes(trossen_arm.Mode.position)
        elif mode == "external_effort":
            self.driver.set_all_modes(trossen_arm.Mode.external_effort)
        elif mode == "idle":
            self.driver.set_all_modes(trossen_arm.Mode.idle)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    
    def print_all_info(self) -> None:
        try:
            pose_base = self.get_p()
            pose_cam = self.get_p_in_cam()
            joints = self.get_j()
            efforts = self.get_efforts()
            temps = self.get_temperatures()
            gripper_pos, gripper_effort = self.get_gripper()
            
            modes = self.driver.get_modes()
            mode_names = [m.name for m in modes]
        except Exception as e:
            print(f"║ [Error] Failed to fetch telemetry: {e}")
            return

        print("\n" + "╔" + "═"*78 + "╗")
        print(f"║ [RobotArmTrossen] STATUS REPORT - IP: {self.cfg.ip:<15}                ║")
        print("╠" + "═"*78 + "╣")


        print(f"║ [EE CARTESIAN POSE]")
        print(f"║   ee 6d pose in Base Frame (m, rad):  X:{pose_base[0]:>7.4f} Y:{pose_base[1]:>7.4f} Z:{pose_base[2]:>7.4f}")
        print(f"║                         RX:{pose_base[3]:>7.3f} RY:{pose_base[4]:>7.3f} RZ:{pose_base[5]:>7.3f}")
        print(f"║   ee 6d pose in Cam Frame  (m, rad):  X:{pose_cam[0]:>7.4f} Y:{pose_cam[1]:>7.4f} Z:{pose_cam[2]:>7.4f}")
        print(f"║                         RX:{pose_cam[3]:>7.3f} RY:{pose_cam[4]:>7.3f} RZ:{pose_cam[5]:>7.3f}")
        
        print("╟" + "─"*78 + "╢")

        print(f"║ [JOINT TELEMETRY] (7 is gripper)")
        print(f"║  ID | Mode      | Angle(rad) | Angle(deg) | Effort(Nm) | Temp(°C)")
        print(f"║ ----|-----------|------------|------------|------------|---------")
        for i in range(len(joints)):
            deg = np.degrees(joints[i])
            print(f"║  {i+1:<2} | {mode_names[i]:<9} | {joints[i]:>10.4f} | {deg:>10.2f} | {efforts[i]:>10.3f} | {temps[i]:>7.1f}")

        print("╟" + "─"*78 + "╢")

        gripper_pct = self.get_gripper_q() * 100
        bar_len = 20
        filled = int(bar_len * self.get_gripper_q())
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"║ [GRIPPER STATE]")
        print(f"║   Position: {gripper_pos:.4f} m [{bar}] {gripper_pct:>5.1f}%")
        print(f"║   Effort:   {gripper_effort:>10.3f} N  | Mode: {mode_names[-1]}")


        print("╟" + "─"*78 + "╢")
        print(f"║ [SYSTEM CONFIG]")
        print(f"║   Draggable: {str(self._is_draggable):<10} | Model: {self.cfg.model_type:<10}")
        print(f"║   EE Type:   {self.cfg.ee_type:<10} | Joints: { self.driver.get_num_joints()} (include gripper)")
        print(f'║   EE Flange Tool: {self.cfg.t_flange_tool[0]:>7.4f}, {self.cfg.t_flange_tool[1]:>7.4f}, {self.cfg.t_flange_tool[2]:>7.4f}, {self.cfg.t_flange_tool[3]:>7.4f}, {self.cfg.t_flange_tool[4]:>7.4f}, {self.cfg.t_flange_tool[5]:>7.4f}')
        
        print("╚" + "═"*78 + "╝\n")


    def close(self) -> None:
        try:
            # self.go_eventual()
            self.driver.set_all_modes(trossen_arm.Mode.idle)
            print("║ [RobotArmTrossen] Connection Closed Safely.")
        except:
            pass


    def move_p_orig(self, pose: Union[List[float], np.ndarray], duration: float = 3.0, blocking: bool = True) -> bool:
        """ pose: [x, y, z, ax, ay, az] """
        try:
            self.driver.set_cartesian_positions(
                list(pose),
                trossen_arm.InterpolationSpace.cartesian,
                duration,
                blocking
            )
            return True
        except Exception as e:
            print(f"║ [Move Error] Cartesian move failed: {e}")
            return False


    def move_p(
        self,
        pose: Union[List[float], np.ndarray],
        duration: float = 3.0,
        blocking: bool = True,
        use_joint_interp: bool = False,
        safe_z_min: float = -0.120
    ) -> bool:
        curr_p = self.get_p()
        try:
            pose = np.array(pose, dtype=np.float64)

            t_target = pose[:3]
            r_target = pose[3:]
            R_target = R.from_rotvec(r_target).as_matrix()
            T_target = np.eye(4)
            T_target[:3, :3] = R_target
            T_target[:3, 3] = t_target

            R_adj = self._get_correction_matrix()
            T_raw_target = T_target @ np.linalg.inv(R_adj)

            t_raw = T_raw_target[:3, 3]

            if t_raw[2] < safe_z_min:
                print(f"║ [Safety] Z-Axis Clamped: {t_raw[2]:.4f} -> {safe_z_min}")
                t_raw[2] = safe_z_min

            r_raw = R.from_matrix(T_raw_target[:3, :3]).as_rotvec()
            pose_raw = np.concatenate([t_raw, r_raw])

            # fix_rotation: every cycle, read the arm's CURRENT rotation from the
            # driver and send it straight back.  The driver sees "target rot ==
            # current rot" → zero rotational motion.  No drift, no oscillation.
            if self._fix_rotation:
                pose_raw[3:] = self.get_p_orig()[3:].copy()

            interp_mode = (trossen_arm.InterpolationSpace.joint
                       if use_joint_interp
                       else trossen_arm.InterpolationSpace.cartesian)

            self.driver.set_cartesian_positions(
                list(pose_raw),
                interp_mode,
                duration,
                blocking
            )

            # print(f"║ 📢📢📢curr_p: {curr_p} goal_p: {pose_raw}")
            return True

        except Exception as e:
            error_msg = str(e)
            
            # 1. Distinguish between mathematical IK failure and actual hardware failure
            if "inverse kinematics" in error_msg.lower() or "singularity" in error_msg.lower():
                print(f"║ ⚠️ [IK Warning] Pose unreachable or singularity hit! Rejecting command.")
                # Return False immediately. DO NOT recover connection! 
                # This allows the InferenceController to execute the "Singularity Escape" anti-windup logic.
                return False
                
            else:
                # 2. Only perform connection recovery for critical hardware or TCP/UDP drops
                print(f"║ 🔴 [Hardware Error] Critical communication drop: {error_msg}")
                detail = self.driver.get_error_information()
                print(f"║ 🔴 [Detailed Info] {detail}")
                
                self.recover_connection()
                return False


    def move_j(self, joints: Union[List[float], np.ndarray], duration: float = 3.0, blocking: bool = True) -> bool:
        """ joints: [j1, j2, j3, j4, j5, j6, gripper_pos] """
        try:
            self.driver.set_all_positions(list(joints), duration, blocking)
            return True
        except Exception as e:
            print(f"║ [Move Error] Joint move failed: {e}")
            return False


    def go_home(self, duration: float = 3.0, blocking: bool = True) -> None:
        home_data = self.cfg.get("home_joints", [0, 0, 0, 0, 0, 0, 0])
        if len(home_data) > 0 and isinstance(home_data[0], (list, tuple, np.ndarray)):
            print(f"║ [RobotArmTrossen] Executing home SEQUENCE ({len(home_data)} steps)...")
            for i, step_joints in enumerate(home_data):
                print(f"║   Step {i+1}/{len(home_data)}: {step_joints}")
                self.move_j(step_joints, duration=duration, blocking=blocking)
        else:
            print(f"║ [RobotArmTrossen] Moving to home pose...")
            self.move_j(home_data, duration=duration, blocking=blocking)

    
    def go_initial(self, duration: float = 3.0, blocking: bool = True) -> None:
        initial_data = self.cfg.get("initial_joints", [0, 0, 0, 0, 0, 0, 0])
        if len(initial_data) > 0 and isinstance(initial_data[0], (list, tuple, np.ndarray)):
            print(f"║ [RobotArmTrossen] Executing initial SEQUENCE ({len(initial_data)} steps)...")
            for i, step_joints in enumerate(initial_data):
                print(f"║   Step {i+1}/{len(initial_data)}: {step_joints}")
                self.move_j(step_joints, duration=duration, blocking=blocking)
        else:
            print(f"║ [RobotArmTrossen] Moving to INITIAL pose...")
            self.move_j(initial_data, duration=duration, blocking=blocking)


    def go_eventual(self, duration: float = 3.0, blocking: bool = True) -> None:
        eventual_joints = self.cfg.get("eventual_joints", [0, 0, 0, 0, 0, 0, 0])
        self.move_j(eventual_joints, duration=duration, blocking=blocking)


    def _get_correction_matrix(self):
        """
        Define coordinate frame correction matrix: rotate 90° around X, then 90° around Y.
        Resulting axis mapping: X_new = Z_old, Y_new = X_old, Z_new = Y_old
        """
        T_align1 = np.array([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        T_align = T_align1
        return T_align


    def get_p_orig(self) -> np.ndarray:
        try:
            # [x, y, z, ax, ay, az] (Rotation vector representation)
            p = np.array(list(self.driver.get_cartesian_positions()))
            self._last_valid_p_orig = p
            return p
    
        except Exception as e:
            print(f"║ ⚠️ [Read Warning] Telemetry lost, using cached position.")
            if hasattr(self, '_last_valid_p_orig'):
                return self._last_valid_p_orig
            return np.zeros(6)


    def get_p(self) -> np.ndarray:
        try:
            p_raw = self.get_p_orig()
            t_raw = p_raw[:3]
            r_raw = p_raw[3:]
            R_raw = R.from_rotvec(r_raw).as_matrix()
            
            T_raw = np.eye(4)
            T_raw[:3, :3] = R_raw
            T_raw[:3, 3] = t_raw
            
            T_new = T_raw @ self._get_correction_matrix()
            
            t_new = T_new[:3, 3]
            r_new = R.from_matrix(T_new[:3, :3]).as_rotvec()
        
            return np.concatenate([t_new, r_new])
        
        except Exception as e:
            if hasattr(self, '_last_valid_p'):
                return self._last_valid_p
            return np.zeros(6)
        
    def get_j(self) -> np.ndarray:
        return np.array(list(self.driver.get_all_positions()))

    def get_efforts(self) -> np.ndarray:
        out = self.driver.get_robot_output()
        return np.array(list(out.joint.all.efforts))

    def get_temperatures(self) -> List[float]:
        out = self.driver.get_robot_output()
        return list(out.joint.all.rotor_temperatures)

    # --- Parameter Settings ---

    def set_speed_limit(self, fraction: float) -> None:
        # Trossen currently controls speed mainly via the duration param in move functions.
        # Store a global scaling factor for internal move functions to reference.
        self._speed_fraction = np.clip(fraction, 0.1, 1.0)

    def set_joints_limit(self, limits: Dict[int, Tuple[float, float]]) -> None:
        # Convert and call the driver's set_joint_limits
        curr_limits = self.driver.get_joint_limits()
        for idx, (low, high) in limits.items():
            if 0 <= idx < len(curr_limits):
                curr_limits[idx].position_min = low
                curr_limits[idx].position_max = high
        self.driver.set_joint_limits(curr_limits)

    # --- Gripper Control ---


    def set_gripper(self, pos: float, effort: Optional[float] = None, blocking: bool = True) -> None:
        if effort is not None:
            self.driver.set_gripper_mode(trossen_arm.Mode.external_effort)
            # Negative value = closing grip
            self.driver.set_gripper_external_effort(-abs(effort), 0.5, blocking)
        else:
            self.driver.set_gripper_mode(trossen_arm.Mode.position)
            # Clamp within physical travel range
            target = np.clip(pos, 0, self._gripper_max)
            self.driver.set_gripper_position(target, 1.0, blocking)


    def get_gripper(self) -> Tuple[float, float]:
        out = self.driver.get_robot_output()
        return out.joint.gripper.position, out.joint.gripper.effort


    def get_gripper_q(self) -> float:
        gripper_pos, _ = self.get_gripper()
        gripper_q = np.clip(gripper_pos / self._gripper_max, 0.0, 1.0)
        return gripper_q

    def close_gripper(self, blocking: bool = True, goal_time: float = 1.0):
        # self.driver.set_gripper_mode(trossen_arm.Mode.position)
        # self.driver.set_arm_modes(trossen_arm.Mode.idle)
        self.driver.set_gripper_position(0, goal_time, blocking)

    def open_gripper(self, blocking: bool = True, goal_time: float = 1.0):
        # self.driver.set_gripper_mode(trossen_arm.Mode.position)
        # self.driver.set_arm_modes(trossen_arm.Mode.idle)
        self.driver.set_gripper_position(self._gripper_max, goal_time, blocking)

    # --- Mode Switching ---

    def set_draggable(self, enable: bool) -> None:
        self._is_draggable = enable
        if enable:
            self.driver.set_all_modes(trossen_arm.Mode.external_effort)
            self.driver.set_all_external_efforts(np.zeros(7), 0.0, False)
            print("║ [RobotArmTrossen] ALL joints are now DRAGGABLE")
        else:
            self.driver.set_arm_modes(trossen_arm.Mode.position)
            self.driver.set_gripper_mode(trossen_arm.Mode.position)
            print("║ [RobotArmTrossen] ALL joints are now LOCKED")


    def update_gravity_comp(self):
        if self._is_draggable:
            out = self.driver.get_robot_output()
            comp = [-e for e in out.joint.arm.compensation_efforts]
            self.driver.set_arm_external_efforts(comp, 0.0, False)
            self.driver.set_gripper_external_effort(0.0, 0.0, False)


    # --- Camera Frame Logic ---

    def get_T_ee_in_base(self) -> np.ndarray:
        p_in_base = self.get_p()
        T_ee_in_base = np.eye(4)
        T_ee_in_base[:3, :3] = R.from_rotvec(p_in_base[3:6]).as_matrix()
        T_ee_in_base[:3, 3] = p_in_base[:3]
        return T_ee_in_base

    @property
    def T_base_in_cam(self) -> np.ndarray:
        return self._T_base_in_cam

    def p_in_cam_to_p_in_base(self, p_in_cam: Union[List[float], np.ndarray]) -> np.ndarray:
        p_in_cam = np.array(p_in_cam)
        T_cam_in_base = np.linalg.inv(self.T_base_in_cam)
        if len(p_in_cam) == 3:
            p_homo_in_cam = np.append(p_in_cam, 1.0)
            p_in_base_homo = T_cam_in_base @ p_homo_in_cam
            p_in_base = p_in_base_homo[:3]
        else:
            T_ee_in_cam = np.eye(4)
            T_ee_in_cam[:3, :3] = R.from_rotvec(p_in_cam[3:6]).as_matrix()
            T_ee_in_cam[:3, 3] = p_in_cam[:3]
            T_ee_in_base = T_cam_in_base @ T_ee_in_cam
            t_in_base = T_ee_in_base[:3, 3]
            r_in_base = R.from_matrix(T_ee_in_base[:3, :3]).as_rotvec()
            p_in_base = np.concatenate([t_in_base, r_in_base])
            
        return p_in_base

    def p_in_base_to_p_in_cam(self, p_in_base: Union[List[float], np.ndarray]) -> np.ndarray:
        p_in_base = np.array(p_in_base)
        if len(p_in_base) == 3:
            p_homo_in_base = np.append(p_in_base, 1.0)
            p_in_cam_homo = self.T_base_in_cam @ p_homo_in_base
            p_in_cam = p_in_cam_homo[:3]
        else:
            T_ee_in_base = np.eye(4)
            T_ee_in_base[:3, :3] = R.from_rotvec(p_in_base[3:6]).as_matrix()
            T_ee_in_base[:3, 3] = p_in_base[:3]
            T_ee_in_cam = self.T_base_in_cam @ T_ee_in_base

            t_in_cam = T_ee_in_cam[:3, 3]
            r_in_cam = R.from_matrix(T_ee_in_cam[:3, :3]).as_rotvec()
            p_in_cam = np.concatenate([t_in_cam, r_in_cam])
        return p_in_cam


    def T_to_p(self, T: np.ndarray) -> np.ndarray:
        t = T[:3, 3]
        r = R.from_matrix(T[:3, :3]).as_rotvec()
        return np.concatenate([t, r])

    def p_to_T(self, p: Union[List[float], np.ndarray]) -> np.ndarray:
        p = np.array(p)
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec(p[3:6]).as_matrix()
        T[:3, 3] = p[:3]
        return T

    def move_p_in_cam(self, p_in_cam: np.ndarray, duration: float = 3.0, blocking: bool = True, use_joint_interp: bool = False, safe_z_min: float = -0.120) -> bool:
        p_in_base = self.p_in_cam_to_p_in_base(p_in_cam)
        return self.move_p(p_in_base, duration, blocking, use_joint_interp, safe_z_min)


    def get_p_in_cam(self) -> np.ndarray:
        return self.p_in_base_to_p_in_cam(self.get_p())


    def get_T_ee_in_cam(self) -> np.ndarray:
        T_ee_in_base = self.get_T_ee_in_base()
        T_ee_in_cam = self.T_base_in_cam @ T_ee_in_base
        return T_ee_in_cam


    def get_vis_p_in_cam(self, img: np.ndarray, k: np.ndarray, d: np.ndarray, p_cam: np.ndarray = None) -> np.ndarray:
        if p_cam is None:
            p_cam = self.get_p_in_cam()
        rvec = p_cam[3:6].reshape(3, 1)
        tvec = p_cam[0:3].reshape(3, 1)
        
        axis_len = 0.1 
        if p_cam[2] > 0:
            cv2.drawFrameAxes(img, k, d, rvec, tvec, axis_len)
            # Define 4 key points in local frame: origin, X tip, Y tip, Z tip
            axis_points_3d = np.float32([
                [0, 0, 0],          # Origin
                [axis_len, 0, 0],   # X axis tip
                [0, axis_len, 0],   # Y axis tip
                [0, 0, axis_len]    # Z axis tip
            ])
            img_pts, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, k, d)
            img_pts = img_pts.astype(int).reshape(-1, 2)
            origin = tuple(img_pts[0])
            pt_x = tuple(img_pts[1])
            pt_y = tuple(img_pts[2])
            pt_z = tuple(img_pts[3])
            # Draw center point
            cv2.circle(img, origin, 5, (255, 255, 255), -1)
            # Draw axis labels (colors in BGR)
            cv2.putText(img, 'X', pt_x, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, 'Y', pt_y, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, 'Z', pt_z, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            # Debug: print projected points (Z-axis may point away from screen)
            # print(f"Origin: {origin}, X: {pt_x}, Y: {pt_y}, Z: {pt_z}")
        return img

        
    def realtime_control(self):
        import tkinter as tk
        from tkinter import ttk
        import threading
        import re
        import numpy as np

        # --- Visual Theme (Apple Dark Mode Style) ---
        C_BG = "#070707"        # Pure black background
        C_CARD = "#243836"      # Dark gray card
        C_BORDER = "#3A3A3C"    # Border color
        C_ACCENT = "#2C7D38"    # Apple green
        C_BLUE = "#0053A6"      # Apple blue
        C_ORANGE = "#C27603"    # Apple orange
        C_TEXT = "#FFFFFF"      
        C_DIM_TEXT = "#8E8E93"  

        # --- Font definitions ---
        # Tkinter does not support font lists; use the most universal Apple-like font.
        # Falls back to Helvetica or Arial if SF Pro is not installed.
        def get_best_font(size, weight="normal"):
            return ("Helvetica", size, weight)

        FONT_UI = get_best_font(11)
        FONT_BOLD = get_best_font(12, "bold")
        FONT_TITLE = get_best_font(18, "bold")
        FONT_MONO = ("Consolas", 10) # Monospace font required for data alignment

        JOINT_LIMITS = {
            0: (-3.14159, 3.14159), 1: (-1.5708, 1.5708), 2: (-3.14159, 0.0),
            3: (-3.14159, 3.14159), 4: (-1.5708, 1.5708), 5: (-3.14159, 3.14159)
        }

        class AppleRoboticTerminal:
            def __init__(self, arm: 'RobotArmTrossen'):
                self.arm = arm
                self.root = tk.Tk()
                self.root.title(f"TROSSEN INTELLIGENT TERMINAL")
                self.root.geometry("1350x950")
                self.root.configure(bg=C_BG)
                
                self.ready = False
                self.auto_update = True
                
                self._create_layout()
                self._sync_all_to_ui()
                
                # Start background threads
                threading.Thread(target=self._telemetry_worker, daemon=True).start()
                threading.Thread(target=self._cli_worker, daemon=True).start()
                
                self.ready = True

            def _create_layout(self):
                # 1. Header Section
                header = tk.Frame(self.root, bg=C_BG, height=80)
                header.pack(fill='x', padx=30, pady=(20, 10))
                
                tk.Label(header, text="●  TROSSEN ROBOTIC SYSTEM v2.5", fg=C_ACCENT, bg=C_BG, font=FONT_TITLE).pack(side='left')
                self.status_tag = tk.Label(header, text="SYSTEM READY", bg=C_BLUE, fg="white", font=FONT_BOLD, width=15, pady=6)
                self.status_tag.pack(side='right')

                # Main container
                container = tk.Frame(self.root, bg=C_BG)
                container.pack(fill='both', expand=True, padx=30, pady=10)

                # 2. Left Panel: Full Telemetry Dashboard
                # Fixed width 450 px
                left_panel = tk.Frame(container, bg=C_BG, width=450)
                left_panel.pack(side='left', fill='both', padx=(0, 20))
                left_panel.pack_propagate(False) 

                self.tele_vars = {
                    "sys": tk.StringVar(), 
                    "pose_base": tk.StringVar(),
                    "pose_cam": tk.StringVar(), 
                    "joint_table": tk.StringVar(), 
                    "gripper": tk.StringVar()
                }

                def create_dashboard_card(title, var, height):
                    f = tk.Frame(left_panel, bg=C_CARD, highlightthickness=1, highlightbackground=C_BORDER)
                    f.pack(fill='x', pady=(0, 15))
                    tk.Label(f, text=title, bg=C_CARD, fg=C_DIM_TEXT, font=get_best_font(10, "bold")).pack(anchor='w', padx=15, pady=(8,0))
                    lbl = tk.Label(f, textvariable=var, bg=C_CARD, fg=C_TEXT, font=FONT_MONO, justify='left', padx=15, pady=10, height=height)
                    lbl.pack(anchor='w')

                create_dashboard_card("SYSTEM CONFIGURATION", self.tele_vars["sys"], 4)
                create_dashboard_card("END-EFFECTOR [BASE FRAME]", self.tele_vars["pose_base"], 2)
                create_dashboard_card("END-EFFECTOR [CAMERA FRAME]", self.tele_vars["pose_cam"], 2)
                create_dashboard_card("JOINT ANALYTICS (Rad | Deg | Nm | Temp)", self.tele_vars["joint_table"], 9)
                create_dashboard_card("GRIPPER REAL-TIME FEEDBACK", self.tele_vars["gripper"], 2)

                # 3. Right Panel: Control Interface
                right_panel = tk.Frame(container, bg=C_BG)
                right_panel.pack(side='right', fill='both', expand=True)

                # --- Joint Space Control ---
                j_group = tk.LabelFrame(right_panel, text=" JOINT SPACE CONTROL ", bg=C_BG, fg=C_DIM_TEXT, font=FONT_BOLD, labelanchor='nw', padx=15, pady=15)
                j_group.pack(fill='x', pady=(0, 20))
                
                self.j_sliders, self.j_entries = [], []
                for i in range(6):
                    row = tk.Frame(j_group, bg=C_BG)
                    row.pack(fill='x', pady=3)
                    tk.Label(row, text=f"J{i+1}", bg=C_BG, fg=C_ACCENT, width=4, font=FONT_BOLD, anchor='w').pack(side='left')
                    low, high = JOINT_LIMITS[i]
                    s = tk.Scale(row, from_=low, to=high, resolution=0.001, orient='horizontal', bg=C_BG, fg="white", 
                                 highlightthickness=0, troughcolor="#2C2C2E", showvalue=0, command=lambda v, idx=i: self._move_req('j', idx, v))
                    s.pack(side='left', fill='x', expand=True, padx=10)
                    e = tk.Entry(row, width=10, bg=C_CARD, fg=C_ACCENT, font=FONT_MONO, bd=0, highlightthickness=1, highlightbackground=C_BORDER)
                    e.pack(side='right')
                    e.bind('<Return>', lambda ev, idx=i: self._entry_confirm('j', idx))
                    self.j_sliders.append(s); self.j_entries.append(e)

                # --- Cartesian Space Control ---
                p_group = tk.LabelFrame(right_panel, text=" CARTESIAN POSE CONTROL ", bg=C_BG, fg=C_DIM_TEXT, font=FONT_BOLD, labelanchor='nw', padx=15, pady=15)
                p_group.pack(fill='x', pady=(0, 20))
                
                self.p_sliders, self.p_entries = [], []
                p_labels = ['X', 'Y', 'Z', 'AX', 'AY', 'AZ']
                for i in range(6):
                    row = tk.Frame(p_group, bg=C_BG)
                    row.pack(fill='x', pady=3)
                    tk.Label(row, text=p_labels[i], bg=C_BG, fg=C_BLUE, width=4, font=FONT_BOLD, anchor='w').pack(side='left')
                    s = tk.Scale(row, from_=-0.8 if i<3 else -3.14, to=0.8 if i<3 else 3.14, resolution=0.001, orient='horizontal',
                                 bg=C_BG, fg="white", highlightthickness=0, troughcolor="#2C2C2E", showvalue=0, command=lambda v, idx=i: self._move_req('p', idx, v))
                    s.pack(side='left', fill='x', expand=True, padx=10)
                    e = tk.Entry(row, width=10, bg=C_CARD, fg=C_BLUE, font=FONT_MONO, bd=0, highlightthickness=1, highlightbackground=C_BORDER)
                    e.pack(side='right')
                    e.bind('<Return>', lambda ev, idx=i: self._entry_confirm('p', idx))
                    self.p_sliders.append(s); self.p_entries.append(e)

                # --- Gripper Control Panel ---
                g_group = tk.LabelFrame(right_panel, text=" GRIPPER ACTUATOR ", bg=C_BG, fg=C_DIM_TEXT, font=FONT_BOLD, labelanchor='nw', padx=15, pady=15)
                g_group.pack(fill='x')
                g_row = tk.Frame(g_group, bg=C_BG)
                g_row.pack(fill='x')
                
                self.g_slider = tk.Scale(g_row, from_=0, to=100, orient='horizontal', bg=C_BG, fg=C_TEXT, 
                                         label="OPEN PERCENTAGE (%)", font=FONT_MONO, highlightthickness=0, troughcolor="#2C2C2E", command=self._grip_scroll)
                self.g_slider.pack(side='left', fill='x', expand=True, padx=(0, 30))
                
                g_btns = tk.Frame(g_row, bg=C_BG)
                g_btns.pack(side='right')
                tk.Button(g_btns, text="OPEN", font=FONT_BOLD, width=8, bg="#333", fg=C_TEXT, command=lambda: self.arm.set_gripper(1.0)).grid(row=0, column=0, padx=2, pady=2)
                tk.Button(g_btns, text="CLOSE", font=FONT_BOLD, width=8, bg="#333", fg=C_TEXT, command=lambda: self.arm.set_gripper(0.0)).grid(row=0, column=1, padx=2, pady=2)
                self.f_input = tk.Entry(g_btns, width=8, bg=C_CARD, fg=C_ORANGE, font=FONT_MONO, bd=0, highlightthickness=1, highlightbackground=C_BORDER)
                self.f_input.grid(row=1, column=0, pady=5); self.f_input.insert(0, "20.0")
                tk.Button(g_btns, text="FORCE(N)", font=get_best_font(9, "bold"), command=self._apply_force_req).grid(row=1, column=1)

                # 4. Footer: Global Controls
                footer = tk.Frame(self.root, bg=C_BG)
                footer.pack(fill='x', side='bottom', padx=30, pady=30)
                
                btn_opt = {"font": FONT_BOLD, "width": 18, "pady": 12, "bd": 0, "cursor": "hand2"}
                tk.Button(footer, text="DRAGGABLE", bg=C_ORANGE, fg="black", **btn_opt, command=self._on_drag_click).pack(side='left', padx=5)
                tk.Button(footer, text="LOCK / SYNC", bg="#3A3A3C", fg=C_BLUE, **btn_opt, command=self._on_lock_click).pack(side='left', padx=5)
                tk.Button(footer, text="GO HOME", bg="#3A3A3C", fg="white", **btn_opt, command=self.arm.go_home).pack(side='left', padx=5)
                tk.Button(footer, text="GO INITIAL", bg="#3A3A3C", fg=C_ACCENT, **btn_opt, command=self.arm.go_initial).pack(side='left', padx=5)
                tk.Button(footer, text="GO ENVENTUAL", bg="#3A3A3C", fg=C_ACCENT, **btn_opt, command=self.arm.go_eventual).pack(side='left', padx=5)
                tk.Button(footer, text="SAFE EXIT", bg="#FF3B30", fg="white", **btn_opt, command=self._on_exit_click).pack(side='right', padx=5)

            # --- Core Control Logic ---
            def _sync_all_to_ui(self):
                """Synchronize all UI controls with the arm's current state."""
                j = self.arm.get_j()
                p = self.arm.get_p()
                for i in range(6):
                    self.j_sliders[i].set(j[i])
                    self.j_entries[i].delete(0, tk.END); self.j_entries[i].insert(0, f"{j[i]:.4f}")
                    self.p_sliders[i].set(p[i])
                    self.p_entries[i].delete(0, tk.END); self.p_entries[i].insert(0, f"{p[i]:.4f}")
                g_pos, _ = self.arm.get_gripper()
                self.g_slider.set((g_pos / self.arm._gripper_max) * 100)

            def _move_req(self, mode, idx, val):
                if not self.ready or self.arm._is_draggable: return
                try:
                    if mode == 'j':
                        target = self.arm.get_j()
                        target[idx] = float(val)
                        self.arm.move_j(target, duration=3.0, blocking=False)
                    else:
                        target = self.arm.get_p()
                        target[idx] = float(val)
                        self.arm.move_p(target, duration=3.0, blocking=False)
                except Exception:
                    # On failure: don't update state or raise, wait for next command
                    pass

            def _entry_confirm(self, mode, idx):
                try:
                    val = float(self.j_entries[idx].get() if mode=='j' else self.p_entries[idx].get())
                    # Send command first; only sync slider if no exception
                    self._move_req(mode, idx, val)
                    if mode=='j': self.j_sliders[idx].set(val)
                    else: self.p_sliders[idx].set(val)
                except Exception:
                    pass

            def _grip_scroll(self, val):
                if not self.ready: return
                try:
                    self.arm.set_gripper((float(val)/100.0)*self.arm._gripper_max, blocking=False)
                except Exception:
                    pass

            def _apply_force_req(self):
                try: 
                    self.arm.set_gripper(0, effort=float(self.f_input.get()))
                except Exception: 
                    pass

            def _on_drag_click(self):
                self.arm.set_draggable(True)
                self.status_tag.config(text="DRAGGABLE", bg=C_ORANGE, fg="black")

            def _on_lock_click(self):
                self.arm.set_draggable(False)
                self.status_tag.config(text="LOCKED", bg=C_BLUE, fg="white")
                self._sync_all_to_ui()

            def _telemetry_worker(self):
                while self.auto_update:
                    try:
                        pb, pc = self.arm.get_p(), self.arm.get_p_in_cam()
                        j, e, t = self.arm.get_j(), self.arm.get_efforts(), self.arm.get_temperatures()
                        gp, ge = self.arm.get_gripper()
                        modes = [m.name for m in self.arm.driver.get_modes()]
                        
                        # System info
                        self.tele_vars["sys"].set(f"IP: {self.arm.cfg.ip}\nMODE: {modes[0].upper()}\nMODEL: {self.arm.cfg.model_type}\nDRAGGABLE: {self.arm._is_draggable}")
                        
                        # Pose info
                        self.tele_vars["pose_base"].set(f"X:{pb[0]:.3f} Y:{pb[1]:.3f} Z:{pb[2]:.3f}\nAX:{pb[3]:.3f} AY:{pb[4]:.3f} AZ:{pb[5]:.3f}")
                        self.tele_vars["pose_cam"].set(f"X:{pc[0]:.3f} Y:{pc[1]:.3f} Z:{pc[2]:.3f}\nAX:{pc[3]:.3f} AY:{pc[4]:.3f} AZ:{pc[5]:.3f}")
                        
                        # Joint table
                        table = ""
                        for i in range(len(j)):
                            name = f"J{i+1}" if i < 6 else "GRP"
                            row_color = "!" if t[i] > 55 else " "
                            table += f"{name:<3} | {j[i]:>6.3f} | {np.degrees(j[i]):>6.1f}° | {e[i]:>5.1f}Nm | {row_color}{int(t[i])}C\n"
                        self.tele_vars["joint_table"].set(table)
                        
                        # Gripper
                        self.tele_vars["gripper"].set(f"POSITION: {gp:.4f} m\nEFFORT:   {ge:.2f} N")
                    except: 
                        import traceback
                        traceback.print_exc()
                    time.sleep(0.05)

            def _cli_worker(self):
                """
                Advanced CLI control engine - independently parses all commands.
                Supported commands:
                - System: d(drag), l(lock), h(home), q(quit)
                - Gripper: o(open), c(close), 50(percentage), f 20(force)
                - Pose: x+0.1, z=0.4, rx-0.2 (mixed axis input supported)
                - Absolute: 0.2 0 0.3 0 1.57 0 (6 numbers)
                """
                # Axis name mapping table
                axis_map = {
                    'x': 0, 'y': 1, 'z': 2,
                    'rx': 3, 'ax': 3,
                    'ry': 4, 'ay': 4,
                    'rz': 5, 'az': 5
                }

                # Print CLI welcome banner
                print("\n" + "═" * 70)
                print("║ \033[95mINTELLIGENT CLI TERMINAL ACTIVE\033[0m".ljust(79) + "║")
                print("║ " + "─" * 66 + " ║")
                print("║  \033[94m[POSE]\033[0m  x+0.1, z=0.5, rx-0.2 | 0.2 0 0.4 0 1.57 0".ljust(81) + "║")
                print("║  \033[92m[GRIP]\033[0m  o (open), c (close), 50 (50%), f 30 (30N force)".ljust(81) + "║")
                print("║  \033[93m[SYST]\033[0m  d (drag), l (lock), h (home), i (initial), e (eventual) q (exit)".ljust(81) + "║")
                print("═" * 70)

                while self.auto_update:
                    try:
                        # Styled prompt: Robot@Trossen >>
                        raw_input = input("\n\033[92mRobot\033[0m@\033[94mTrossen\033[0m >> ").strip().lower()
                        if not raw_input: continue
                        if raw_input == 'q': 
                            print("\033[91m>> Initiating Safe Shutdown...\033[0m")
                            self._on_exit_click()
                            break

                        # 1. Handle system-level shortcut commands
                        if raw_input == 'd':
                            self._on_drag_click()
                            print("\033[93m>> Mode: DRAGGABLE (Gravity Comp Active)\033[0m")
                            continue
                        if raw_input == 'l':
                            self._on_lock_click()
                            print("\033[94m>> Mode: LOCKED / SYNCHRONIZED\033[0m")
                            continue
                        if raw_input == 'h':
                            print(">> Moving to HOME position (3.0s)...")
                            self.arm.go_home()
                            continue

                        if raw_input == 'i':
                            print(">> Executing INITIAL joint sequence...")
                            self.arm.go_initial()
                            continue
                        
                        if raw_input == 'e':
                            print(">> Executing EVENTUAL joint sequence...")
                            self.arm.go_eventual()
                            continue
                    
                        # 2. Handle gripper shortcuts and numeric control
                        if raw_input == 'o':
                            self.arm.set_gripper(1.0, blocking=False)
                            print(">> Gripper: OPENING")
                            continue
                        if raw_input == 'c':
                            self.arm.set_gripper(0.0, blocking=False)
                            print(">> Gripper: CLOSING")
                            continue
                        
                        # Gripper force control (e.g., f 20)
                        if raw_input.startswith('f'):
                            parts = raw_input.split()
                            if len(parts) > 1:
                                f_val = float(parts[1])
                                self.arm.set_gripper(0, effort=f_val, blocking=False)
                                print(f"\033[93m>> Gripper: Applying {f_val}N force\033[0m")
                            continue

                        # Gripper percentage control (e.g., 80)
                        if raw_input.isdigit():
                            pct = int(raw_input)
                            if 0 <= pct <= 100:
                                target_g = (pct / 100.0) * self.arm._gripper_max
                                self.arm.set_gripper(target_g, blocking=False)
                                print(f">> Gripper: Set to {pct}% ({target_g:.4f}m)")
                            continue

                        # 3. Handle full 6D pose input (6 numbers)
                        nums = re.findall(r'[-+]?\d*\.\d+|\d+', raw_input)
                        if len(nums) == 6 and not any(op in raw_input for op in ['+', '-', '=']):
                            target = [float(x) for x in nums]
                            print(f">> Cartesian: Moving to absolute target...")
                            self.arm.move_p(target, duration=3.0, blocking=False)
                            continue

                        # 4. Advanced parsing: mixed axis commands (x+0.1, z=0.5, rx=0)
                        # Regex match: letter(axis) + operator(=+-) + number
                        axis_cmds = re.findall(r'([a-zA-Z]+)([+=-])([-+]?[0-9.]+)', raw_input.replace(" ", ""))
                        if axis_cmds:
                            current_p = list(self.arm.get_p())
                            modified_p = current_p.copy()
                            valid_move = False
                            
                            for axis_name, op, val_str in axis_cmds:
                                if axis_name in axis_map:
                                    idx = axis_map[axis_name]
                                    val = float(val_str)
                                    if op == '=':
                                        modified_p[idx] = val
                                        print(f"   [SET] {axis_name.upper()} = {val}")
                                    elif op == '+':
                                        modified_p[idx] += val
                                        print(f"   [ADD] {axis_name.upper()} + {val}")
                                    elif op == '-':
                                        modified_p[idx] -= val
                                        print(f"   [SUB] {axis_name.upper()} - {val}")
                                    valid_move = True
                            
                            if valid_move:
                                # Send command to the robot arm
                                self.arm.move_p(modified_p, duration=3.0, blocking=False)
                            continue

                        print("\033[91m>> Unknown Command. Use d, l, h, o, c, f <val>, or axis ops (e.g. x+0.1)\033[0m")

                    except Exception:
                        # Silently handle any error (e.g., IK failure) and wait for next command
                        pass
            

            def _on_exit_click(self):
                self.auto_update = False
                self.arm.close()
                self.root.destroy()

        app = AppleRoboticTerminal(self)
        app.root.mainloop()


    def jog_xyz_forever(self, step: float = 0.02, duration: float = 1.0):
        self.set_mode('position')
        print(f"[jog_xyz_forever] step={step}m, duration={duration}s. Ctrl+C to exit.")
        try:
            while True:
                p = self.get_p() 
                target = np.array(p, dtype=np.float64)
                # target[0:6] += step
                target[0] += step
                ok = self.move_p(target, duration=duration, blocking=False)
                
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\n[jog_xyz_forever] user exit (Ctrl+C).")


# ---------------------------------------------------------------------------
# Pure teleop helpers (no recording)
# ---------------------------------------------------------------------------

def _teleop_kb_listener(evt_quit: threading.Event,
                        evt_emergency: threading.Event) -> None:
    """Lightweight keyboard listener: 'q' quits, 'e' toggles emergency stop."""
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not evt_quit.is_set():
            ch = sys.stdin.read(1).upper()
            if ch == 'Q':
                evt_quit.set()
                return
            elif ch == 'E':
                evt_emergency.set()
    except Exception:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _compose_dashboard(views: Dict[str, np.ndarray],
                       panel_height: int = 720) -> np.ndarray:
    """Stack camera frames horizontally into a single image (no overlays)."""
    panels = []
    for img in views.values():
        if img is None:
            continue
        h, w = img.shape[:2]
        new_w = max(1, int(w * panel_height / h))
        panels.append(cv2.resize(img, (new_w, panel_height)))
    if not panels:
        return np.zeros((panel_height, panel_height, 3), dtype=np.uint8)
    return np.hstack(panels)


def run_teleop(leaders: Dict[str, 'RobotArmTrossen'],
               followers: Dict[str, 'RobotArmTrossen'],
               control_freq: float = 200.0,
               ema_alpha: float = 0.5,
               feedback_gain: float = 0.001,
               cameras: Optional[Dict[str, object]] = None,
               visualize: bool = False,
               fullscreen: bool = True) -> None:
    """Run pure leader→follower teleop (no recording) on one or two arm pairs.

    Press 'q' (or Ctrl+C) to quit. Press 'e' to toggle emergency stop.

    Parameters
    ----------
    leaders, followers : dict[str, RobotArmTrossen]
        Same keys (e.g. {"right": ...} or {"right": ..., "left": ...}).
    cameras : optional dict[str, CamRS]
        Cameras to display when ``visualize=True``. Keys whose name contains
        "right"/"left" get an EE-axis overlay for the matching follower.
    """
    assert set(leaders.keys()) == set(followers.keys()), \
        "leaders and followers must have matching keys"

    print("[run_teleop] Moving all arms to home position...")
    for arm in {**leaders, **followers}.values():
        arm.go_home(duration=3.0, blocking=True)

    for name, leader in leaders.items():
        leader.set_draggable(True)
        print(f"[run_teleop] Leader '{name}' -> draggable mode.")

    evt_quit = threading.Event()
    evt_emergency = threading.Event()
    state = {"emergency": False,
             "ema_pos": {n: None for n in leaders}}

    def teleop_loop():
        dt = 1.0 / control_freq
        print(f"[run_teleop] Teleop loop @ {control_freq} Hz, EMA alpha={ema_alpha}")
        while not evt_quit.is_set():
            t0 = time.monotonic()
            if state["emergency"]:
                time.sleep(dt)
                continue
            try:
                for name, leader in leaders.items():
                    follower = followers[name]
                    pos = np.array(leader.driver.get_all_positions())
                    vel = leader.driver.get_all_velocities()
                    prev = state["ema_pos"][name]
                    if prev is None:
                        smoothed = pos.copy()
                    else:
                        smoothed = ema_alpha * pos + (1.0 - ema_alpha) * prev
                    state["ema_pos"][name] = smoothed
                    follower.driver.set_all_positions(smoothed.tolist(), 0.0, False, vel)
                    eff = follower.driver.get_all_external_efforts()
                    leader.driver.set_all_external_efforts(
                        -feedback_gain * np.array(eff), 0.0, False)
            except Exception as e:
                print(f"[run_teleop] loop error: {e}")
            elapsed = time.monotonic() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        print("[run_teleop] Teleop loop stopped.")

    def handle_emergency():
        if not state["emergency"]:
            print("[run_teleop] !!! EMERGENCY STOP !!!")
            state["emergency"] = True
            for arm in {**leaders, **followers}.values():
                try:
                    arm.driver.set_all_modes(trossen_arm.Mode.position)
                except Exception:
                    pass
        else:
            print("[run_teleop] Recovering from emergency stop...")
            for leader in leaders.values():
                try:
                    leader.set_draggable(True)
                except Exception:
                    pass
            state["emergency"] = False

    th_teleop = threading.Thread(target=teleop_loop, daemon=True)
    th_teleop.start()
    th_kb = threading.Thread(
        target=_teleop_kb_listener, args=(evt_quit, evt_emergency), daemon=True)
    th_kb.start()

    print("\n" + "=" * 70)
    print("[run_teleop] System ready. Press 'q' to quit, 'e' for emergency stop.")
    print("=" * 70 + "\n")

    try:
        if visualize and cameras:
            print(f"[run_teleop] Visualization enabled: {list(cameras.keys())}"
                  f" (fullscreen={fullscreen})")
            window_name = "[teleop] dashboard"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            if fullscreen:
                cv2.setWindowProperty(window_name,
                                      cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            while not evt_quit.is_set():
                if evt_emergency.is_set():
                    evt_emergency.clear()
                    handle_emergency()
                views: Dict[str, np.ndarray] = {}
                for cam_id, cam in cameras.items():
                    cdata = cam.get_rgbd()
                    if cdata is None or getattr(cdata, "rgb", None) is None:
                        continue
                    views[cam_id] = cdata.rgb
                if views:
                    composite = _compose_dashboard(views)
                    cv2.imshow(window_name, composite)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    evt_quit.set()
                    break
                elif key == ord('e'):
                    handle_emergency()
            cv2.destroyAllWindows()
        else:
            while not evt_quit.is_set():
                if evt_emergency.is_set():
                    evt_emergency.clear()
                    handle_emergency()
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[run_teleop] KeyboardInterrupt received.")
    finally:
        evt_quit.set()
        th_teleop.join(timeout=3.0)

        for leader in leaders.values():
            try:
                leader.set_draggable(False)
            except Exception:
                pass
        # Reset followers to position mode too — the streaming
        # set_all_positions(..., 0.0, False, vel) commands during teleop
        # leave the driver in a state where a subsequent blocking go_home
        # is silently dropped. Forcing position mode clears that state.
        for follower in followers.values():
            try:
                follower.set_draggable(False)
            except Exception:
                pass

        print("[run_teleop] Returning all arms to home...")
        for name, arm in {**followers, **leaders}.items():
            try:
                arm.go_home(duration=3.0, blocking=True)
                print(f"  [{name}] home ✓")
            except Exception as e:
                print(f"  [{name}] go_home failed: {e}")

        print("[run_teleop] Closing arms...")
        for arm in {**followers, **leaders}.values():
            try:
                arm.close()
            except Exception:
                pass
        if cameras:
            for cam in cameras.values():
                try:
                    cam.close()
                except Exception:
                    pass
        print("[run_teleop] Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trossen arm utility: single-arm GUI control or leader-follower teleop.")
    parser.add_argument("--mode", type=str, default="control",
                        choices=["control", "teleop"],
                        help="control = single-arm GUI (default); teleop = leader→follower mirroring")
    parser.add_argument("--side", type=str, default="right",
                        choices=["right", "left", "right_leader", "left_leader", "both"],
                        help=("control mode: pick any of the 4 arms "
                              "(right|left|right_leader|left_leader). "
                              "teleop mode: right|left|both."))
    parser.add_argument("--visualize", action="store_true",
                        help="Show all cameras in one composite window during teleop.")
    parser.add_argument("--no_fullscreen", action="store_true",
                        help="Disable fullscreen for the visualization window "
                             "(default: fullscreen).")

    parser.add_argument("--follower_right", type=str,
                        default="./cfg/hardware/RobotArmTrossenRight.yaml")
    parser.add_argument("--follower_left", type=str,
                        default="./cfg/hardware/RobotArmTrossenLeft.yaml")
    parser.add_argument("--leader_right", type=str,
                        default="./cfg/hardware/RobotArmTrossenRightLeader.yaml")
    parser.add_argument("--leader_left", type=str,
                        default="./cfg/hardware/RobotArmTrossenLeftLeader.yaml")
    parser.add_argument("--cam_top", type=str,
                        default="./cfg/hardware/CamRS.yaml")
    parser.add_argument("--cam_wrist_right", type=str,
                        default="./cfg/hardware/CamRSWristRight.yaml")
    parser.add_argument("--cam_wrist_left", type=str,
                        default="./cfg/hardware/CamRSWristLeft.yaml")

    parser.add_argument("--control_freq", type=float, default=200.0)
    parser.add_argument("--ema_alpha", type=float, default=0.5)
    parser.add_argument("--feedback_gain", type=float, default=0.001)
    args = parser.parse_args()

    if args.mode == "control":
        if args.side == "both":
            print("[ERROR] --mode control supports only a single arm; "
                  "--side cannot be 'both'.")
            sys.exit(1)
        side_to_cfg = {
            "right":        args.follower_right,
            "left":         args.follower_left,
            "right_leader": args.leader_right,
            "left_leader":  args.leader_left,
        }
        arm = RobotArmTrossen(side_to_cfg[args.side])
        arm.realtime_control()
    else:  # teleop
        if args.side in ("right_leader", "left_leader"):
            print("[ERROR] --mode teleop expects --side right|left|both.")
            sys.exit(1)
        sides = ["right", "left"] if args.side == "both" else [args.side]

        leaders: Dict[str, RobotArmTrossen] = {}
        followers: Dict[str, RobotArmTrossen] = {}
        for s in sides:
            if s == "right":
                leaders["right"]   = RobotArmTrossen(args.leader_right)
                followers["right"] = RobotArmTrossen(args.follower_right)
            else:
                leaders["left"]    = RobotArmTrossen(args.leader_left)
                followers["left"]  = RobotArmTrossen(args.follower_left)

        cameras: Dict[str, object] = {}
        if args.visualize:
            from inference.CamRS import CamRS
            cameras["top"] = CamRS(args.cam_top)
            for s in sides:
                cam_path = args.cam_wrist_right if s == "right" else args.cam_wrist_left
                cameras[f"wrist_{s}"] = CamRS(cam_path)

        run_teleop(leaders, followers,
                   control_freq=args.control_freq,
                   ema_alpha=args.ema_alpha,
                   feedback_gain=args.feedback_gain,
                   cameras=cameras if args.visualize else None,
                   visualize=args.visualize,
                   fullscreen=not args.no_fullscreen)

# Examples:
#   GUI control of any single arm (4 choices):
#     python -m inference.RobotArmTrossen --side right
#     python -m inference.RobotArmTrossen --side left
#     python -m inference.RobotArmTrossen --side right_leader
#     python -m inference.RobotArmTrossen --side left_leader
#   Pure teleop (no recording):
#     python -m inference.RobotArmTrossen --mode teleop --side right
#     python -m inference.RobotArmTrossen --mode teleop --side both
#   Pure teleop with camera visualization:
#     python -m inference.RobotArmTrossen --mode teleop --side both --visualize