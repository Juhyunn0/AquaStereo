#!/usr/bin/env python3
"""Run AquaStereo on a Luxonis OAK-D W (or any OAK stereo device).

The OAK rectifies the wide-FOV mono pair on-device and streams the rectified
frames to the host; AquaStereo then predicts disparity, which is turned into
metric depth with the device's own calibration.

    python oak_aquastereo.py --restore_ckpt checkpoints/AquaStereo_vits_best.pth
    python oak_aquastereo.py --source images --left a.png --right b.png

Keys in the live window: q quit, s save frame, p save point cloud,
space pause, d toggle disparity/depth colouring, [ ] change refinement iters.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.AquaStereo import AquaStereo
from core.utils.utils import InputPadder


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

# Values the AquaStereo module reads off its args namespace; they mirror the
# defaults in evaluate_stereo.py and must match how the checkpoint was trained.
MODEL_DEFAULTS = dict(
    hidden_dims=[128, 128, 128],
    corr_levels=2,
    corr_radius=4,
    n_downsample=2,
    n_gru_layers=3,
    max_disp=768,
    s_disp_range=48,
    m_disp_range=96,
    l_disp_range=192,
    s_disp_interval=1,
    m_disp_interval=2,
    l_disp_interval=4,
    num_perception_frame=2,
)


def _strip_wrappers(key):
    while True:
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        else:
            return key


def _extract_state(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for name in ("model", "model_state", "state_dict"):
        if name in ckpt:
            return ckpt[name]
    return ckpt


def _resize_perception_frames(state, model):
    """Match the checkpoint's learnable perception frames to this model's size."""
    ref = model.state_dict()
    for key in list(state):
        if not key.endswith("encoder.perception_frames") or key not in ref:
            continue
        src, dst = state[key], ref[key]
        if src.shape == dst.shape:
            continue
        _, c, t, h0, w0 = src.shape
        th, tw = dst.shape[-2:]
        flat = src.permute(0, 2, 1, 3, 4).reshape(-1, c, h0, w0).float()
        flat = torch.nn.functional.interpolate(flat, size=(th, tw), mode="bilinear", align_corners=False)
        state[key] = flat.reshape(1, t, c, th, tw).permute(0, 2, 1, 3, 4).to(dtype=dst.dtype)
    return state


def build_model(ckpt_path, vit_size, device, mixed_precision=False, precision_dtype="float32"):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    args = argparse.Namespace(
        **MODEL_DEFAULTS,
        vit_size=vit_size,
        # Set so the encoder skips loading the external DINOv2/X3D pretrains -
        # both backbones live inside the AquaStereo checkpoint.
        restore_ckpt=ckpt_path,
        mixed_precision=mixed_precision,
        precision_dtype=precision_dtype,
    )
    model = AquaStereo(args)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = _extract_state(ckpt)
    known = set(model.state_dict())
    filtered = {}
    dropped = 0
    for key, value in state.items():
        clean = _strip_wrappers(key)
        if clean in known:
            filtered[clean] = value
        else:
            dropped += 1
    filtered = _resize_perception_frames(filtered, model)
    model.load_state_dict(filtered, strict=True)
    if dropped:
        print(f"[model] ignored {dropped} checkpoint keys not used by this model")

    model.to(device).eval()
    return model


@torch.no_grad()
def predict_disparity(model, left_rgb, right_rgb, iters, device, mixed_precision=False):
    """left_rgb / right_rgb: HxWx3 uint8 RGB. Returns HxW float32 disparity in px."""
    def to_tensor(img):
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
        return t[None].to(device, non_blocking=True)

    # AquaStereo normalises internally, so it wants raw 0..255 values.
    left, right = to_tensor(left_rgb), to_tensor(right_rgb)
    padder = InputPadder(left.shape, divis_by=32)
    left, right = padder.pad(left, right)

    with torch.amp.autocast("cuda", enabled=mixed_precision and device.type == "cuda"):
        disp = model(left, right, iters=iters, test_mode=True)

    disp = padder.unpad(disp.float())
    return disp[0, 0].cpu().numpy()


# ---------------------------------------------------------------------------
# frame sources
# ---------------------------------------------------------------------------

class OakSource:
    """Rectified stereo pair + calibration from an OAK device."""

    def __init__(self, width, height, fps, alpha=None, preset="DEFAULT"):
        import depthai as dai
        self.dai = dai

        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            raise RuntimeError(
                "No OAK device found. Plug the camera into a USB3 port and check "
                "`lsusb | grep 03e7`. If it only shows up as a bootloader, try a "
                "different cable - USB2 charge-only cables do not carry data."
            )
        print(f"[oak] found {len(devices)} device(s): " +
              ", ".join(f"{d.getMxId()} ({d.protocol.name}, {d.state.name})" for d in devices))

        self.pipeline = dai.Pipeline()
        left_cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right_cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        left_out = left_cam.requestOutput((width, height), fps=fps)
        right_out = right_cam.requestOutput((width, height), fps=fps)

        stereo = self.pipeline.create(dai.node.StereoDepth).build(
            left_out, right_out,
            presetMode=getattr(dai.node.StereoDepth.PresetMode, preset),
        )
        stereo.setRectification(True)
        # The OAK-D W's ~127 deg lenses need mesh rectification rather than a
        # 3x3 homography; AUTO already picks mesh above 85 deg, this makes it explicit.
        stereo.enableDistortionCorrection(True)
        if alpha is not None:
            stereo.setAlphaScaling(alpha)
        self.alpha = alpha

        self.q_left = stereo.rectifiedLeft.createOutputQueue(maxSize=4, blocking=False)
        self.q_right = stereo.rectifiedRight.createOutputQueue(maxSize=4, blocking=False)

        device = self.pipeline.getDefaultDevice()
        calib = device.readCalibration()
        # DepthAI rectifies both views into the RIGHT camera's intrinsics
        # (H_left = K_right * R_rect * inv(K_left)), so CAM_C gives the
        # rectified focal length.
        k = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, width, height))
        self.fx = float(k[0, 0])
        self.fy = float(k[1, 1])
        self.cx = float(k[0, 2])
        self.cy = float(k[1, 2])
        self.baseline_m = abs(float(calib.getBaselineDistance(
            useSpecTranslation=False, unit=dai.LengthUnit.METER)))
        if self.baseline_m <= 0:
            self.baseline_m = abs(float(calib.getBaselineDistance(
                useSpecTranslation=True, unit=dai.LengthUnit.METER)))
            print("[oak] calibrated baseline unavailable; falling back to board spec")

        try:
            model_b = calib.getDistortionModel(dai.CameraBoardSocket.CAM_B).name
            fov_b = calib.getFov(dai.CameraBoardSocket.CAM_B)
            print(f"[oak] left lens: {model_b} model, {fov_b:.1f} deg HFOV")
        except Exception:
            pass

        self.width, self.height = width, height
        self._intrinsics_checked = False
        self.pipeline.start()
        print(f"[oak] rectified stream {width}x{height} @ {fps} fps | "
              f"fx={self.fx:.2f} px  baseline={self.baseline_m * 100:.2f} cm")

    def _check_frame_intrinsics(self, frame):
        """Alpha scaling changes the effective camera matrix; trust the frame if it disagrees."""
        self._intrinsics_checked = True
        try:
            k = np.array(frame.getTransformation().getIntrinsicMatrix())
        except Exception:
            return
        fx = float(k[0, 0])
        if fx > 0 and abs(fx - self.fx) / self.fx > 0.02:
            print(f"[oak] per-frame fx={fx:.2f} differs from calibration fx={self.fx:.2f}; "
                  f"using the per-frame value (alpha scaling in effect)")
            self.fx, self.fy = fx, float(k[1, 1])
            self.cx, self.cy = float(k[0, 2]), float(k[1, 2])

    def read(self):
        left = self.q_left.get()
        right = self.q_right.get()
        # rectifiedLeft/Right come out of one processing pass, but a dropped
        # frame on a non-blocking queue can still skew them.
        for _ in range(8):
            if left.getSequenceNum() == right.getSequenceNum():
                break
            if left.getSequenceNum() < right.getSequenceNum():
                left = self.q_left.get()
            else:
                right = self.q_right.get()
        if not self._intrinsics_checked:
            self._check_frame_intrinsics(left)
        return left.getCvFrame(), right.getCvFrame()

    def is_running(self):
        return self.pipeline.isRunning()

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


class ImagePairSource:
    """A single rectified pair from disk - lets you test without the camera."""

    def __init__(self, left_path, right_path, fx=None, baseline_m=None):
        left = cv2.imread(left_path, cv2.IMREAD_UNCHANGED)
        right = cv2.imread(right_path, cv2.IMREAD_UNCHANGED)
        if left is None or right is None:
            raise FileNotFoundError(f"could not read {left_path} / {right_path}")
        if left.shape[:2] != right.shape[:2]:
            raise ValueError(f"left {left.shape[:2]} and right {right.shape[:2]} differ in size")
        self.left, self.right = left, right
        self.height, self.width = left.shape[:2]
        self.fx = fx if fx else 0.8 * self.width
        self.baseline_m = baseline_m if baseline_m else 0.075
        self.cx, self.cy = self.width / 2.0, self.height / 2.0
        self.fy = self.fx
        if fx is None or baseline_m is None:
            print("[images] no --fx/--baseline given; depth values are indicative only")

    def read(self):
        return self.left, self.right

    def is_running(self):
        return True

    def close(self):
        pass


# ---------------------------------------------------------------------------
# visualisation / export
# ---------------------------------------------------------------------------

def to_rgb(frame):
    """Whatever the source hands us -> HxWx3 uint8 RGB."""
    if frame.dtype != np.uint8:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def colorize(values, lo, hi, cmap=cv2.COLORMAP_TURBO, invalid=None):
    norm = np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    if invalid is not None:
        img[invalid] = 0
    return img


def label(img, text, y):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_ply(path, disp, rgb, fx, fy, cx, cy, baseline_m, min_disp, max_depth):
    valid = disp > min_disp
    z = np.zeros_like(disp)
    z[valid] = fx * baseline_m / disp[valid]
    valid &= (z > 0) & (z < max_depth)

    ys, xs = np.nonzero(valid)
    zs = z[valid]
    pts = np.stack([(xs - cx) * zs / fx, (ys - cy) * zs / fy, zs], axis=1)
    cols = rgb[valid]

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, zz), (r, g, b) in zip(pts, cols):
            f.write(f"{x:.4f} {y:.4f} {zz:.4f} {r} {g} {b}\n")
    return len(pts)


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AquaStereo disparity/depth from an OAK-D W",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--restore_ckpt", default="checkpoints/AquaStereo_vits_best.pth",
                   help="AquaStereo checkpoint (.pth)")
    p.add_argument("--vit_size", choices=["vits", "vitb"], default=None,
                   help="DINOv2 backbone size; inferred from the checkpoint filename if omitted")
    p.add_argument("--iters", type=int, default=16,
                   help="GRU refinement iterations; higher is sharper but slower")
    p.add_argument("--mixed_precision", action="store_true",
                   help="run the fp16 autocast path (roughly 1.5x faster)")

    p.add_argument("--source", choices=["oak", "images"], default="oak")
    p.add_argument("--left", help="left image (--source images)")
    p.add_argument("--right", help="right image (--source images)")

    p.add_argument("--width", type=int, default=640, help="stream width (OV9282 native is 1280)")
    p.add_argument("--height", type=int, default=400, help="stream height (native 800)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--preset", default="DEFAULT",
                   help="StereoDepth preset: DEFAULT, ACCURACY, DENSITY, ROBOTICS, HIGH_DETAIL, FAST_ACCURACY, FAST_DENSITY, FACE")
    p.add_argument("--alpha", type=float, default=None,
                   help="rectification alpha scaling: 0 crops to valid pixels, 1 keeps all "
                        "source pixels (black wide-lens borders). Unset = device default")

    p.add_argument("--fx", type=float, default=None, help="override focal length in px")
    p.add_argument("--baseline", type=float, default=None, help="override baseline in metres")
    p.add_argument("--water_n", type=float, default=1.0,
                   help="refractive index for an underwater FLAT port (water ~1.333). "
                        "Scales the effective focal length. Leave at 1.0 in air or behind a "
                        "correctly centred dome port")

    p.add_argument("--max_depth", type=float, default=10.0, help="depth colour/point-cloud clip in metres")
    p.add_argument("--min_disp", type=float, default=0.5, help="disparities below this are treated as invalid")
    p.add_argument("--vis_max_disp", type=float, default=None,
                   help="fix the disparity colour scale; default auto-scales per frame")
    p.add_argument("--view", choices=["disparity", "depth"], default="disparity")
    p.add_argument("--out_dir", default="oak_output")
    p.add_argument("--no_display", action="store_true", help="headless: process and save, no window")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = until 'q')")
    return p.parse_args()


def main():
    args = parse_args()

    vit_size = args.vit_size
    if vit_size is None:
        name = os.path.basename(args.restore_ckpt).lower()
        vit_size = "vitb" if "vitb" in name else "vits"
        print(f"[model] inferred --vit_size {vit_size} from the checkpoint name")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[model] WARNING: no CUDA device; inference will take many seconds per frame")

    print(f"[model] loading {args.restore_ckpt} ({vit_size}) on {device}")
    model = build_model(args.restore_ckpt, vit_size, device, args.mixed_precision)

    if args.source == "images":
        if not (args.left and args.right):
            sys.exit("--source images requires --left and --right")
        source = ImagePairSource(args.left, args.right, args.fx, args.baseline)
    else:
        source = OakSource(args.width, args.height, args.fps, args.alpha, args.preset)

    fx = args.fx if args.fx else source.fx
    baseline_m = args.baseline if args.baseline else source.baseline_m
    # Behind a flat port the effective focal length grows by roughly the
    # refractive index of water; the baseline is unaffected.
    fx_eff = fx * args.water_n
    if args.water_n != 1.0:
        print(f"[geom] flat-port correction n={args.water_n}: fx {fx:.2f} -> {fx_eff:.2f} px")
    print(f"[geom] depth = {fx_eff:.2f} * {baseline_m:.4f} / disparity  "
          f"(1 px disparity ~ {fx_eff * baseline_m:.2f} m)")

    os.makedirs(args.out_dir, exist_ok=True)

    # Warm up so the first timed frame is not dominated by CUDA/cuDNN setup.
    warm_l, warm_r = source.read()
    warm_l, warm_r = to_rgb(warm_l), to_rgb(warm_r)
    predict_disparity(model, warm_l, warm_r, args.iters, device, args.mixed_precision)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[model] warm-up done on a {warm_l.shape[1]}x{warm_l.shape[0]} pair")

    static = args.source == "images"
    state = {"iters": args.iters, "view": args.view, "paused": False, "hover": None}
    window = "AquaStereo | OAK-D W"

    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        def on_mouse(event, x, y, flags, _):
            state["hover"] = (x, y)
        cv2.setMouseCallback(window, on_mouse)

    frame_idx = 0
    saved = 0
    ema_ms = None
    left_rgb, right_rgb, disp = warm_l, warm_r, None

    try:
        while source.is_running():
            if not state["paused"] or disp is None:
                if disp is not None:  # first pass reuses the warm-up frames
                    left_raw, right_raw = source.read()
                    left_rgb, right_rgb = to_rgb(left_raw), to_rgb(right_raw)

                t0 = time.time()
                disp = predict_disparity(model, left_rgb, right_rgb, state["iters"],
                                         device, args.mixed_precision)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ms = (time.time() - t0) * 1000
                ema_ms = ms if ema_ms is None else 0.9 * ema_ms + 0.1 * ms
                frame_idx += 1
                if static:
                    # One pair from disk: keep the window interactive, stop re-running.
                    state["paused"] = True

            invalid = disp <= args.min_disp
            depth = np.zeros_like(disp)
            np.divide(fx_eff * baseline_m, disp, out=depth, where=~invalid)

            if state["view"] == "depth":
                panel = colorize(np.clip(depth, 0, args.max_depth), 0, args.max_depth,
                                 cv2.COLORMAP_INFERNO, invalid)
                scale_txt = f"depth 0-{args.max_depth:.1f} m"
            else:
                hi = args.vis_max_disp or max(float(np.percentile(disp[~invalid], 99)) if (~invalid).any() else 1.0, 1.0)
                panel = colorize(disp, 0, hi, cv2.COLORMAP_TURBO, invalid)
                scale_txt = f"disparity 0-{hi:.1f} px"

            if args.no_display:
                if static or (args.frames and frame_idx >= args.frames):
                    break
                continue

            left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
            canvas = np.hstack([left_bgr, panel])

            label(canvas, f"{ema_ms:6.1f} ms  ({1000 / max(ema_ms, 1e-6):4.1f} fps)   "
                          f"iters={state['iters']}  {vit_size}", 22)
            label(canvas, f"{scale_txt}   frame {frame_idx}"
                          + ("   [PAUSED]" if state["paused"] else ""), 44)

            if state["hover"]:
                hx, hy = state["hover"]
                # The canvas is left image | disparity panel, so fold the x back.
                px = hx % left_bgr.shape[1]
                if 0 <= hy < disp.shape[0] and 0 <= px < disp.shape[1]:
                    d = float(disp[hy, px])
                    z = float(depth[hy, px])
                    txt = f"({px},{hy})  d={d:6.2f} px  z={z:6.3f} m" if d > args.min_disp \
                        else f"({px},{hy})  no disparity"
                    label(canvas, txt, canvas.shape[0] - 14)

            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                state["paused"] = not state["paused"]
            elif key == ord("d"):
                state["view"] = "depth" if state["view"] == "disparity" else "disparity"
            elif key == ord("["):
                state["iters"] = max(1, state["iters"] - 4)
            elif key == ord("]"):
                state["iters"] = min(64, state["iters"] + 4)
            elif key == ord("s"):
                stem = os.path.join(args.out_dir, f"frame_{saved:04d}")
                cv2.imwrite(f"{stem}_left.png", left_bgr)
                cv2.imwrite(f"{stem}_right.png", cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"{stem}_vis.png", panel)
                np.save(f"{stem}_disparity.npy", disp)
                np.save(f"{stem}_depth.npy", depth)
                print(f"[save] {stem}_*.png / .npy")
                saved += 1
            elif key == ord("p"):
                path = os.path.join(args.out_dir, f"cloud_{saved:04d}.ply")
                n = save_ply(path, disp, left_rgb, fx_eff, fx_eff, source.cx, source.cy,
                             baseline_m, args.min_disp, args.max_depth)
                print(f"[save] {path} ({n} points)")

            if args.frames and frame_idx >= args.frames:
                break
    except KeyboardInterrupt:
        print("\n[main] interrupted")
    finally:
        source.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    if (static or args.no_display) and disp is not None:
        stem = os.path.join(args.out_dir, "result")
        cv2.imwrite(f"{stem}_vis.png", panel)
        np.save(f"{stem}_disparity.npy", disp)
        np.save(f"{stem}_depth.npy", depth)
        valid = disp > args.min_disp
        if valid.any():
            print(f"[result] disparity {disp[valid].min():.2f}-{disp[valid].max():.2f} px, "
                  f"depth {depth[valid].min():.3f}-{depth[valid].max():.3f} m")
        print(f"[result] wrote {stem}_vis.png, {stem}_disparity.npy, {stem}_depth.npy")


if __name__ == "__main__":
    main()
