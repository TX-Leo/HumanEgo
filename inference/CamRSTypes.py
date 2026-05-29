import os
import cv2
import json
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List


def _apply_color_to_depth(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    z_min = 0.15
    z_max = 1.50
    depth_clipped = np.clip(img, z_min, z_max)
    depth_norm = (depth_clipped - z_min) / (z_max - z_min)
    depth_8u = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)
    invalid_mask = (img <= 0) | np.isnan(img)
    depth_color[invalid_mask] = 0
    return depth_color


@dataclass
class CamRSData:
    idx: int = 0 
    ts: int = 0             # timestamp in ns
    rgb: np.ndarray = None
    depth_m: np.ndarray = None
    rgb_raw: np.ndarray = None
    depth_raw_m: np.ndarray = None

    @property
    def depth_color(self) -> np.ndarray:
        return _apply_color_to_depth(self.depth_m)
    
    @property
    def depth_raw_color(self) -> np.ndarray:
        return _apply_color_to_depth(self.depth_raw_m)

    def get_data(self):
        return self.idx, self.ts, self.rgb, self.depth_m, self.depth_color, self.rgb_raw, self.depth_raw_m, self.depth_raw_color

    def save_data(self, save_dir: str = None):
        os.makedirs(save_dir, exist_ok=True)
        prefix = f"{self.idx:04d}_{self.ts}"

        cv2.imwrite(os.path.join(save_dir, f"rgb_{prefix}.png"), self.rgb)
        np.save(os.path.join(save_dir, f"depth_m_{prefix}.npy"), self.depth_m)
        cv2.imwrite(os.path.join(save_dir, f"depth_color_{prefix}.png"), self.depth_color)
        
        cv2.imwrite(os.path.join(save_dir, f"rgb_raw_{prefix}.png"), self.rgb_raw)
        np.save(os.path.join(save_dir, f"depth_raw_m_{prefix}.npy"), self.depth_raw_m)
        cv2.imwrite(os.path.join(save_dir, f"depth_raw_color_{prefix}.png"), self.depth_raw_color)


@dataclass
class CamRSStruct:
    fps: int = 0
    h_raw: int = 0
    w_raw: int = 0
    h: int = 0              # H
    w: int = 0              # W
    k_rgb: np.ndarray = None    # (3, 3) camera intrinsic
    d_rgb: np.ndarray = None    # (1, 5) camera distortion
    k_depth: np.ndarray = None    # (3, 3) camera intrinsic
    d_depth: np.ndarray = None    # (1, 5) camera distortion
    k_rgb_raw: np.ndarray = None    # (3, 3) camera intrinsic
    d_rgb_raw: np.ndarray = None    # (1, 5) camera distortion
    k_depth_raw: np.ndarray = None    # (3, 3) camera intrinsic
    d_depth_raw: np.ndarray = None    # (1, 5) camera distortion
    depth_scale: float = 0.0 # m/unit
    c2o: np.ndarray = None  # (4, 4)
   
    cam: List[CamRSData] = field(default_factory=list)

    
    def __len__(self):
        return len(self.cam)


    @staticmethod
    def _safe_list(arr):
        return arr.tolist() if isinstance(arr, np.ndarray) else arr


    def _save_cam_config_json(self, verbose: str = True):
        save_path = os.path.join("./results", "cam_cfg.json")
        json_data = {
            "fps": self.fps,
            "h_raw": self.h_raw,
            "w_raw": self.w_raw,
            "h": self.h,
            "w": self.w,
            "k_rgb": self._safe_list(self.k_rgb),
            "d_rgb": self._safe_list(self.d_rgb),
            "k_depth": self._safe_list(self.k_depth),
            "d_depth": self._safe_list(self.d_depth),
            "k_rgb_raw": self._safe_list(self.k_rgb_raw),
            "d_rgb_raw": self._safe_list(self.d_rgb_raw),
            "k_depth_raw": self._safe_list(self.k_depth_raw),
            "d_depth_raw": self._safe_list(self.d_depth_raw),
            "depth_scale": self.depth_scale,
        }
        with open(save_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"[***] cam_cfg.json saved to: {save_path}")
        
        if verbose:
            print("╔" + "═" * 78 + "╗")
            print(f"║{'[CamRSStruct] Stream & Intrinsics':^78}║")
            print("╠" + "═" * 78 + "╣")
            print(f"║  Size:  Raw: {self.w_raw}x{self.h_raw}, Final: {self.w}x{self.h} ║")
            print(f"║  Color: \n k_rgb: \n {self.k_rgb} \n d_rgb: {self.d_rgb} ║")
            print(f"║  Depth: \n k_depth: \n {self.k_depth} \n d_depth: {self.d_depth} \n depth_scale: {self.depth_scale:.6f} m/unit ║")
            print(f"║  Raw Color: \n k_rgb_raw: \n {self.k_rgb_raw} \n d_rgb_raw: {self.d_rgb_raw} ║ ")
            print(f"║  Raw Depth: \n k_depth_raw: \n {self.k_depth_raw} \n d_depth_raw: {self.d_depth_raw} ║")
            print("╚" + "═" * 78 + "╝")

    
    def save_history(self):
        if not self.cam:
            print("║ [CamRSStruct] Warning: No data in memory to save.")
            return

        session_id = time.strftime("%Y%m%d_%H%M%S")
        history_dir = os.path.join("./results", session_id)
        os.makedirs(history_dir, exist_ok=True)

        print("╔" + "═" * 78 + "╗")
        print(f"║ [CamRSStruct] Saving history ({len(self.cam)} frames)...{'':^36}║")
        print(f"║ Target: {history_dir:<66} ║")

        for data in self.cam:
            data.save_data(history_dir)
            if data.idx % 5 == 0:
                print(f"║   Progress: frame {data.idx}/{len(self.cam)} saved.{'':^38}║")

        meta_path = os.path.join(history_dir, "session_config.json")
        json_data = {
            "session_ts": session_id,
            "frame_count": len(self.cam),
            "config": {
                "fps": self.fps,
                "h_raw": self.h_raw, 
                "w_raw": self.w_raw,
                "h": self.h, 
                "w": self.w,
                "k_rgb": self._safe_list(self.k_rgb),
                "d_rgb": self._safe_list(self.d_rgb),
                "k_depth": self._safe_list(self.k_depth),
                "d_depth": self._safe_list(self.d_depth),
                "k_rgb_raw": self._safe_list(self.k_rgb_raw),
                "d_rgb_raw": self._safe_list(self.d_rgb_raw),
                "k_depth_raw": self._safe_list(self.k_depth_raw),
                "d_depth_raw": self._safe_list(self.d_depth_raw),
                "depth_scale": self.depth_scale,
            }
        }
        with open(meta_path, 'w') as f:
            json.dump(json_data, f, indent=4)

        print(f"║ [CamRSStruct] Session Meta saved to: {os.path.basename(meta_path):<46} ║")
        print("╚" + "═" * 78 + "╝")
        