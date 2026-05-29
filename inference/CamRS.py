import time
import numpy as np
import cv2
import argparse
import pyrealsense2 as rs
from typing import Any, Dict, List, Optional, Tuple

from inference.CamRSTypes import CamRSStruct, CamRSData

from utils.utils_io import load_cfg

class CamRS():
    def __init__(self, cfg_path: str):
        self.cfg = load_cfg(cfg_path)
        self.init()
        
        
    def init(self):
        try:
            selected_serial = self._select_rs_device(serial=self.cfg.rs_device_serial, verbose=False)

            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_device(selected_serial)

            self.config.enable_stream(rs.stream.color, int(self.cfg.w_raw), int(self.cfg.h_raw), rs.format.bgr8, int(self.cfg.fps))
            self.config.enable_stream(rs.stream.depth, int(self.cfg.w_raw), int(self.cfg.h_raw), rs.format.z16, int(self.cfg.fps))

            self.profile = self.pipeline.start(self.config)

            # Align (depth -> color)
            self.align = rs.align(rs.stream.color)

            # Get intrinsics / scale
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = float(self.depth_sensor.get_depth_scale())

            self.color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()

            self.crop_x0, self.crop_y0, self.crop_w, self.crop_h, self.scale = self._get_crop_parameters()

            self.k_rgb, self.d_rgb, self.k_rgb_raw, self.d_rgb_raw = self._get_intrinsics(self.color_stream)
            self.k_depth, self.d_depth, self.k_depth_raw, self.d_depth_raw  = self._get_intrinsics(self.depth_stream)

            self.cam_struct = CamRSStruct(fps=self.cfg.fps,
                                h_raw=self.cfg.h_raw, w_raw=self.cfg.w_raw,
                                h=self.cfg.h, w=self.cfg.w,
                                k_rgb=self.k_rgb, d_rgb=self.d_rgb,
                                k_depth=self.k_depth, d_depth=self.d_depth,
                                k_rgb_raw=self.k_rgb_raw, d_rgb_raw=self.d_rgb_raw,
                                k_depth_raw=self.k_depth_raw, d_depth_raw=self.d_depth_raw,
                                depth_scale=self.depth_scale,
                                cam=[])
            
            self.cam_struct._save_cam_config_json(verbose=True)
            self.idx = 0
            self._warmup()
            print("║ [CamRS] Init Success")
            return True

        except Exception as e:
            print(f"║ [CamRS] Init Failed: {e}")
            return False
        

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        

    def _get_crop_parameters(self):
        raw_w = float(self.cfg.w_raw)
        raw_h = float(self.cfg.h_raw)
        target_w = float(self.cfg.w)
        target_h = float(self.cfg.h)

        target_aspect = target_w / target_h
        raw_aspect = raw_w / raw_h

        if raw_aspect > target_aspect:
            crop_h = raw_h
            crop_w = raw_h * target_aspect
        else:
            crop_w = raw_w
            crop_h = raw_w / target_aspect

        crop_x0 = int((raw_w - crop_w) // 2)
        crop_y0 = int((raw_h - crop_h) // 2)
        crop_w = int(crop_w)
        crop_h = int(crop_h)

        scale = target_w / crop_w

        return crop_x0, crop_y0, crop_w, crop_h, scale


    def _get_intrinsics(self, stream_profile) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        intrinsics = stream_profile.get_intrinsics()
        k_raw = np.array([[intrinsics.fx, 0.0, intrinsics.ppx],
                      [0.0, intrinsics.fy, intrinsics.ppy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        d_raw = np.array(intrinsics.coeffs)
        k, d = self._update_intrinsics(k_raw, d_raw)
        
        return k,d, k_raw, d_raw


    def _update_intrinsics(self, k_raw: np.ndarray, d_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k_updated = k_raw.copy()
        k_updated[0, 2] = (k_raw[0, 2] - self.crop_x0) * self.scale
        k_updated[1, 2] = (k_raw[1, 2] - self.crop_y0) * self.scale
        k_updated[0, 0] *= self.scale
        k_updated[1, 1] *= self.scale
        d_updated = d_raw.copy()
        print(f"║ [Intrinsic Update] {int(self.cfg.w_raw)}x{int(self.cfg.h_raw)} -> Crop({self.crop_w}x{self.crop_h}) -> Resize({int(self.cfg.w)}x{int(self.cfg.h)})")
        return k_updated, d_updated


    def _select_rs_device(self, serial: str, verbose: bool = True) -> str:
        devices = self.list_all_rs_devices(verbose=verbose)
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices detected.")
        serial = (serial or "").strip()
        if not serial:
            raise RuntimeError("serial is empty.")
        matches = [d for d in devices if d["serial"] == serial]
        if len(matches) != 1:
            raise RuntimeError(f"Serial match failed or ambiguous: serial='{serial}', matches={len(matches)}")
        return matches[0]["serial"]


    def list_all_rs_devices(self, verbose: bool = True):
            ctx = rs.context()
            devs = ctx.query_devices()

            devices_info = []
            if devs.size() == 0:
                if verbose:
                    print("╔══════════════════════════════════════════════════════════════╗")
                    print("║ [CamRS] No RealSense device detected.                        ║")
                    print("╚══════════════════════════════════════════════════════════════╝")
                return devices_info

            for i, dev in enumerate(devs):
                def _get(info_enum):
                    try:
                        return dev.get_info(info_enum)
                    except Exception:
                        return ""

                info = {
                    "index": i,
                    "name": _get(rs.camera_info.name),
                    "serial": _get(rs.camera_info.serial_number),
                    "physical_port": _get(rs.camera_info.physical_port),
                    "usb_type": _get(rs.camera_info.usb_type_descriptor),
                    "firmware": _get(rs.camera_info.firmware_version),
                }
                devices_info.append(info)

            if verbose:
                print("╔══════════════════════════════════════════════════════════════╗")
                print(f"║ [CamRS] RealSense devices detected: {len(devices_info):<3}                         ║")
                print("╠══════════════════════════════════════════════════════════════╣")
                for d in devices_info:
                    idx = d["index"]
                    name = d["name"] or "Unknown"
                    serial = d["serial"] or "Unknown"
                    port = d["physical_port"] or "Unknown"
                    usb = d["usb_type"] or "Unknown"
                    fw = d["firmware"] or "Unknown"
                    print(f"║ [{idx}] {name:<18}  serial={serial:<14}  usb={usb:<6} ║")
                    print(f"║      port={port:<52} ║")
                    print(f"║      fw  ={fw:<52} ║")
                    print("╠══════════════════════════════════════════════════════════════╣")
                print("╚══════════════════════════════════════════════════════════════╝")

            return devices_info


    def _warmup(self):
        n = int(self.cfg.warmup_frames_number)
        if n <= 0:
            return
        for _ in range(n):
            self.get_rgbd_raw()
        print(f"║ [CamRS] Warmup frames: {n}")


    def _crop_img(self, img: np.ndarray) -> np.ndarray:
        return img[self.crop_y0 : self.crop_y0 + self.crop_h, 
                   self.crop_x0 : self.crop_x0 + self.crop_w]


    def _resize_img(self, img: np.ndarray, is_depth: bool = False) -> np.ndarray:
        target_size = (int(self.cfg.w), int(self.cfg.h))
        interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_AREA
        return cv2.resize(img, target_size, interpolation=interp)


    def get_rgbd_raw(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        frames = self.pipeline.wait_for_frames()
        
        if self.align is not None:
            frames = self.align.process(frames)
        
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        
        if not color or not depth:
            return None, None
        
        rgb_raw = np.asanyarray(color.get_data()).copy()   # BGR uint8
        depth_raw = np.asanyarray(depth.get_data()).astype(np.float32)  # uint16 -> float32 units
        depth_raw_m = depth_raw * self.depth_scale

        return rgb_raw, depth_raw_m


    def get_rgbd(self, save_dir: str = None):
        ts = int(time.time() * 1000)

        rgb_raw, depth_raw_m = self.get_rgbd_raw()
        if rgb_raw is None or depth_raw_m is None:
            return None, None, None, None

        rgb_cropped = self._crop_img(rgb_raw)
        depth_cropped_m = self._crop_img(depth_raw_m)

        rgb = self._resize_img(rgb_cropped, is_depth=False)
        depth_m = self._resize_img(depth_cropped_m, is_depth=True)
   
        cam_data = CamRSData(idx=self.idx, ts=ts,
                                rgb=rgb, depth_m=depth_m,
                                rgb_raw=rgb_raw, depth_raw_m=depth_raw_m)
        self.cam_struct.cam.append(cam_data)
        self.idx +=1 
        
        if save_dir:
            cam_data.save_data(save_dir)

        return cam_data
    

    def _robust_mad(self, x: np.ndarray) -> float:
            # median absolute deviation
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            if x.size == 0:
                return 0.0
            med = np.median(x)
            return float(np.median(np.abs(x - med)))


    def _dist_transform_inside(self, mask_u8: np.ndarray) -> np.ndarray:
        # distance to boundary for inside pixels (mask>0), 0 outside
        inside = (mask_u8 > 0).astype(np.uint8)
        return cv2.distanceTransform(inside, distanceType=cv2.DIST_L2, maskSize=5)


    def _median_quality(self, qs: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not qs: return {}
            def med(key, default=np.nan):
                vals = [q.get(key, default) for q in qs if q.get(key, None) is not None]
                vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
                return float(np.median(vals)) if vals else float("nan")
            return {
                "n_valid": med("n_valid"),
                "n_keep": med("n_keep"),
                "mad_depth_m": med("mad_depth_m"),
                "edge_dist_px": med("edge_dist_px"),
            }
    

    def _med_of_q(self, key, qs):
        vals = [q[key] for q in qs if key in q and q[key] is not None]
        return float(np.median(vals)) if vals else 0.0


    def lift3d_for_single_frame(self, uv: Tuple[float, float], depth_m: np.ndarray, mask: Optional[np.ndarray], k: np.ndarray,) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Robustly lift a single 2D pixel to a 3D point (camera frame).
        Returns:
          xyz (3,) or None
          quality dict: n_valid, mad, edge_dist_px, z_hat, status
        """

        u, v = float(uv[0]), float(uv[1])
        h, w = depth_m.shape[:2]
        ui, vi = int(round(u)), int(round(v))
        if ui < 0 or ui >= w or vi < 0 or vi >= h:
            return None, {"valid": False, "status": "oob"}

        r = int(getattr(self.cfg, "lift_roi_radius_px", 4))
        x0, x1 = max(0, ui - r), min(w - 1, ui + r)
        y0, y1 = max(0, vi - r), min(h - 1, vi + r)

        roi = depth_m[y0:y1 + 1, x0:x1 + 1]
        if roi.size == 0:
            return None, {"valid": False, "status": "empty_roi"}

        if mask is not None:
            mroi = mask[y0:y1 + 1, x0:x1 + 1] > 0
            roi = roi[mroi]

        # depth validity
        zmin = float(getattr(self.cfg, "lift_min_depth_m", 0.15))
        zmax = float(getattr(self.cfg, "lift_max_depth_m", 2.50))
        roi = roi[np.isfinite(roi)]
        roi = roi[(roi > zmin) & (roi < zmax)]

        n_valid = int(roi.size)
        if n_valid < int(getattr(self.cfg, "lift_min_valid", 15)):
            return None, {"valid": False, "status": "too_few_depth", "n_valid": n_valid}

        z_med = float(np.median(roi))
        mad = self._robust_mad(roi)
        k_mad = float(getattr(self.cfg, "lift_mad_k", 3.5))
        abs_gate = float(getattr(self.cfg, "lift_abs_gate_m", 0.010))
        gate = k_mad * mad + abs_gate
        keep = np.abs(roi - z_med) <= gate
        roi2 = roi[keep]
        n_keep = int(roi2.size)
        if n_keep < int(getattr(self.cfg, "lift_min_valid", 15)):
            return None, {"valid": False, "status": "gated_out", "n_valid": n_valid, "n_keep": n_keep, "mad": float(mad)}

        z_hat = float(np.median(roi2))

        # edge distance (if mask)
        edge_dist = None
        if mask is not None:
            dt = self._dist_transform_inside((mask > 0).astype(np.uint8) * 255)
            edge_dist = float(dt[vi, ui])

        # backproject
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])
        x = (u - cx) / fx * z_hat
        y = (v - cy) / fy * z_hat
        xyz = np.array([x, y, z_hat], dtype=np.float64)

        summary_q = {
            "valid": True,
            "status": "ok",
            "n_valid": n_valid,
            "n_keep": n_keep,
            "z_hat": z_hat,
            "mad_depth_m": float(mad),
            "edge_dist_px": edge_dist,
        }

        # soft failure if near boundary
        edge_min = float(getattr(self.cfg, "lift_edge_dist_min_px", 0.0))
        if edge_dist is not None and edge_dist < edge_min:
            summary_q["status"] = "near_mask_boundary"

        return xyz, summary_q


    def lift3d_for_multi_frames(self, uv: Tuple[float, float], depth_m_list: List[np.ndarray], mask: Optional[np.ndarray], k: np.ndarray):

        xyzs: List[np.ndarray] = []
        qs: List[Dict[str, Any]] = []

        n_to_process = len(depth_m_list)
        for i in range(n_to_process):
            xyz, q = self.lift3d_for_single_frame(uv, depth_m_list[i], mask, k)
            if xyz is not None:
                xyzs.append(xyz)
                qs.append(q)

        if not xyzs:
            return None, {"valid": False, "status": "no_valid_frames", "n_frames_attempted": n_to_process}

        X = np.stack(xyzs, axis=0)
        xyz = np.median(X, axis=0)
        diffs = np.linalg.norm(X - xyz[None, :], axis=1)
        mad_dispersion = self._robust_mad(diffs)

        q_med_report = self._median_quality(qs)

        summary_q = {
            "valid": True,
            "status": "ok",
            "n_frames_valid": len(xyzs),
            "n_frames_total": n_to_process,
            "dispersion_m_mad": mad_dispersion,
            "z_hat_med": xyz[2],
            "avg_mad_depth_m": self._med_of_q("mad_depth_m", qs),
            "avg_edge_dist_px": self._med_of_q("edge_dist_px", qs),
            "q_median": q_med_report,
        }

        if mad_dispersion > 0.03:
            summary_q["status"] = "unstable_across_frames"

        return xyz, summary_q


    def realtime_show(self, record: bool = False):
        win_name = "CamRS Dashboard (Top: RAW | Bottom: PROCESSED)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("║ [CamRS] Starting realtime show...")
        print("║   - Press 'q' or 'ESC' to quit.")

        disp_h = 500 

        def prepare_display(img, label):
            if img is None:
                return np.zeros((disp_h, disp_h, 3), dtype=np.uint8)
            
            h, w = img.shape[:2]
            aspect = w / h
            new_w = int(disp_h * aspect)
            res = cv2.resize(img, (new_w, disp_h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(res, (0, 0), (220, 35), (0, 0, 0), -1)
            text = f"{label}: {w}x{h}"
            cv2.putText(res, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            return res

        try:
            while True:
                cam_data = self.get_rgbd()
                idx, ts, rgb, depth_m, depth_color, rgb_raw, depth_raw_m, depth_raw_color = cam_data.get_data()
                if rgb_raw is None:
                    continue
                v1 = prepare_display(rgb_raw, "RAW RGB")
                v2 = prepare_display(depth_raw_color, "RAW DEPTH COLOR")
                v3 = prepare_display(rgb, "PROC RGB")
                v4 = prepare_display(depth_color, "PROC DEPTH COLOR")

                row_raw = np.hstack([v1, v2])
                row_proc = np.hstack([v3, v4])
                
                diff = row_raw.shape[1] - row_proc.shape[1]
                if diff > 0:
                    padding = np.zeros((disp_h, diff, 3), dtype=np.uint8)
                    row_proc = np.hstack([row_proc, padding])
                elif diff < 0:
                    padding = np.zeros((disp_h, abs(diff), 3), dtype=np.uint8)
                    row_raw = np.hstack([row_raw, padding])

                canvas = np.vstack([row_raw, row_proc])
                cv2.putText(canvas, f"GLOBAL IDX: {idx:04d} | TS: {ts}", (20, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)
                cv2.imshow(win_name, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: 
                    break
        finally:
            cv2.destroyAllWindows()
            if record:
                self.cam_struct.save_history()
            print("║ [CamRS] Realtime show stopped.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="./cfg/hardware/CamRSWristLeft.yaml")
    args = parser.parse_args()

    cam = CamRS(args.cfg_path)
    # cam.list_all_rs_devices()
    cam.realtime_show()

# python -m inference.CamRS