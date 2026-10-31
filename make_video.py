
# make_video.py was made to film the medication round and save pictures of the ward

#run from the repo folder (after python Build_scene.py)
#what the video shows:
#   main view     a camera that follows the robot from the south (the low cut-away wall),
#   with the planned route drawn on the floor (green), the robot's OWN estimate
#   of where it is (blue ring), and a green sight-line whenever the camera reads a marker
#   top right what the robot's camera sees
#   bottom right  a map: obstacles, the route, the estimate (blue) and the true position (orange)
#   top left what the robot is doing now, and every stop of the round
#the film uses the simulator's true pose only to point the camera and draw the map - exactly like
#the scoring code, it never reaches the robot
#
#needs:  pip install imageio imageio-ffmpeg  (H.264 .mp4 that plays everywhere). without them the
#        video is written with OpenCV's mp4v codec instead, which most players also open

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import frames as F
from Build_scene import OBSTACLES, ROOM_X, ROOM_Y, SCENE_OUT, WARD
from run_delivery import run_delivery, TrialConditions
from Control.nav_settings import MARKER_POSITIONS

MEDIA = Path(__file__).resolve().parent / "media"

INK = (40, 44, 60)
WHITE = (255, 255, 255)
GREEN = (60, 170, 110)
BLUE = (70, 120, 220)
ORANGE = (240, 130, 60)
GREY = (150, 150, 160)

PHASE_TEXT = {"scan": "Scanning the ward with the LiDAR", "align": "Turning to face the route",
              "track": "Driving to {stop}", "deliver": "Delivering medication at {stop}",
              "stopped": "Parked at {stop}"}


def load_font(size, bold=False):

    names = (["DejaVuSans-Bold.ttf", "arialbd.ttf", "segoeuib.ttf", "Arial Bold.ttf"] if bold else
             ["DejaVuSans.ttf", "arial.ttf", "segoeui.ttf", "Arial.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def lerp(a, b, t):
    return a + (b - a) * t


# =====================================================================================#
# ------------------------------------ VIDEO FILE ------------------------------------ #
# =====================================================================================#

class VideoFile:
    def __init__(self, path, fps, size):
        self.path, self.kind = Path(path), None
        try:
            import imageio.v2 as imageio
            self.writer = imageio.get_writer(str(self.path), fps=fps, codec="libx264", pixelformat="yuv420p",
                                             ffmpeg_params=["-preset", "veryfast", "-crf", "20"],
                                             macro_block_size=8)
            self.kind = "h264"
        except Exception:
            w, h = size
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            self.kind = "mp4v"

    def add(self, rgb):
        if self.kind == "h264":
            self.writer.append_data(rgb)
        else:
            self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.kind == "h264":
            self.writer.close()
        else:
            self.writer.release()

# =====================================================================================#
# ------------------------------------- THE FILM ------------------------------------- #
# =====================================================================================#

class Film:
    def __init__(self, out_file, width=1280, height=720, fps=24, speed=3.0, shadows=True):
        self.W, self.H, self.fps, self.speed, self.shadows = width, height, fps, speed, shadows
        self.ticks_per_frame = max(1, int(round(speed / (fps * 0.02))))      #control ticks are 20 ms
        self.out_file = out_file
        self.video = None
        self.tick = 0
        self.frames = 0
        self.m = self.d = None
        self.cam_state = None
        self.last_seen = {}                 #marker id -> clock it was last read
        self.trail = []
        self.font = {k: load_font(s, b) for k, (s, b) in
                     {"title": (26, True), "body": (19, False), "small": (15, False), "chip": (15, True),
                      "card": (22, False), "card_title": (34, True)}.items()}
        self.last_info = None

    # ------------------------------------------------------------------ setup ----------
    def _setup(self, m, d):
        self.m, self.d = m, d
        #anti-aliasing for the film only: the robot's camera keeps the scene's own setting
        old = m.vis.quality.offsamples
        m.vis.quality.offsamples = 4
        m.vis.quality.shadowsize = 2048
        self.main = mujoco.Renderer(m, self.H, self.W)
        m.vis.quality.offsamples = old
        self.inset = mujoco.Renderer(m, 240, 320)
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.sun = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, "sun")
        self.video = VideoFile(self.out_file, self.fps, (self.W, self.H))
        self.has_robot_cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, F.CAMERA_NAME) >= 0

    def true_pose(self):
        p = self.d.xpos[self.base]
        R = self.d.xmat[self.base].reshape(3, 3)
        return np.array([p[0], p[1], math.atan2(R[1, 0], R[0, 0])])

    # ----------------------------------------------------------------- cameras ---------
    def overview_cam(self):
        return {"lookat": np.array([ROOM_X / 2, ROOM_Y / 2 - 0.4, 0.0]), "distance": 1.25 * max(ROOM_X, ROOM_Y) + 1.0,
                "elevation": -52.0, "azimuth": 90.0}

    def follow_cam(self, pose):
        #looks north from the south, over the low wall, gently swinging with the robot's x
        return {"lookat": np.array([pose[0], pose[1] + 0.35, 0.15]), "distance": 4.4, "elevation": -48.0,
                "azimuth": 90.0 + 14.0 * (pose[0] - ROOM_X / 2) / (ROOM_X / 2)}

    def blend(self, a, b, t):
        return {k: lerp(a[k], b[k], t) for k in a}

    def camera(self, params):
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:] = params["lookat"]
        c.distance, c.elevation, c.azimuth = params["distance"], params["elevation"], params["azimuth"]
        return c

    # ---------------------------------------------------------- 3D annotations ---------
    def _add_capsule(self, scn, a, b, width, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
        scn.ngeom += 1

    def _add_disc(self, scn, xy, r, rgba, z=0.006):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([r, r, 0.003]),
                            np.array([xy[0], xy[1], z]), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        scn.ngeom += 1

    def annotate(self, scn, info):
        path = info.get("path")
        if path is not None and len(path) > 1:
            pts = [path[0]]
            for p in path[1:]:
                if np.hypot(*(p - pts[-1])) > 0.08:
                    pts.append(p)
            pts.append(path[-1])
            for a, b in zip(pts[:-1], pts[1:]):
                self._add_capsule(scn, (a[0], a[1], 0.012), (b[0], b[1], 0.012), 0.012, (0.20, 0.72, 0.48, 0.85))
        #where the robot THINKS it is: a blue ring and heading line
        x, y, th = info["ekf_x"]
        for k in range(16):
            a0, a1 = 2 * np.pi * k / 16, 2 * np.pi * (k + 0.6) / 16
            self._add_capsule(scn, (x + 0.27 * np.cos(a0), y + 0.27 * np.sin(a0), 0.02),
                              (x + 0.27 * np.cos(a1), y + 0.27 * np.sin(a1), 0.02), 0.008, (0.25, 0.45, 0.95, 0.9))
        self._add_capsule(scn, (x, y, 0.02), (x + 0.36 * np.cos(th), y + 0.36 * np.sin(th), 0.02), 0.008,
                          (0.25, 0.45, 0.95, 0.9))
        #sight-lines to markers read in the last 0.3 s
        pose = self.true_pose()
        cam = np.array([pose[0] + F.CAMERA_OFFSET[0] * np.cos(pose[2]),
                        pose[1] + F.CAMERA_OFFSET[0] * np.sin(pose[2]), F.CAMERA_OFFSET[2]])
        for marker_id, t_seen in self.last_seen.items():
            if info["clock"] - t_seen < 0.3:
                mx, my = MARKER_POSITIONS[marker_id]
                self._add_capsule(scn, cam, (mx, my, 0.15), 0.005, (0.25, 0.85, 0.45, 0.75))

    # ------------------------------------------------------------- 2D overlays --------
    def minimap(self, info, w=320, h=250):
        img = Image.new("RGB", (w, h), (252, 248, 240))
        dr = ImageDraw.Draw(img)
        pad = 14
        s = min((w - 2 * pad) / ROOM_X, (h - 2 * pad - 14) / ROOM_Y)
        ox, oy = (w - s * ROOM_X) / 2, pad + 14

        def P(x, y):
            return (ox + s * x, oy + s * (ROOM_Y - y))

        dr.text((pad, 4), "Map (blue = estimate, orange = true)", font=self.font["small"], fill=INK)
        dr.rectangle([P(0, ROOM_Y), P(ROOM_X, 0)], outline=(120, 120, 135), width=3)
        for name, x0, y0, x1, y1, *_ in OBSTACLES:
            fill = (205, 215, 235) if name.startswith("bed") else (226, 222, 214)
            if name.startswith("wall"):
                continue
            dr.rectangle([P(x0, y1), P(x1, y0)], fill=fill, outline=(170, 170, 180))
        path = info.get("path")
        if path is not None and len(path) > 1:
            dr.line([P(*p) for p in path], fill=GREEN, width=3)
        if len(self.trail) > 1:
            dr.line([P(*p) for p in self.trail], fill=ORANGE, width=2)
        done = set(info["done"])
        for k, stop in enumerate(info["stops"]):
            px, py = P(*stop["park"][:2])
            colour = GREEN if stop["name"] in done else (ORANGE if k == info["leg"] else GREY)
            dr.ellipse([px - 9, py - 9, px + 9, py + 9], fill=colour)
            dr.text((px, py), str(k + 1), font=self.font["chip"], fill=WHITE, anchor="mm")
        x, y, th = info["ekf_x"]
        cx, cy = P(x, y)
        r = s * F.ROBOT_RADIUS
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=BLUE, width=3)
        dr.line([(cx, cy), (cx + 1.6 * r * np.cos(th), cy - 1.6 * r * np.sin(th))], fill=BLUE, width=3)
        tx, ty = P(*self.true_pose()[:2])
        dr.ellipse([tx - 3, ty - 3, tx + 3, ty + 3], fill=ORANGE)
        return img

    def panel(self, base, box, radius=14, alpha=200):
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rounded_rectangle(box, radius, fill=(255, 255, 255, alpha))
        base.alpha_composite(overlay)

    def compose(self, main_rgb, info, inset_rgb):
        img = Image.fromarray(main_rgb).convert("RGBA")
        W = self.W
        #caption, top left
        stop = info["stop"]["label"]
        phase = PHASE_TEXT.get(info["phase"], info["phase"]).format(stop=stop)
        n, k = len(info["stops"]), info["leg"]
        self.panel(img, (16, 16, 560, 128))
        dr = ImageDraw.Draw(img)
        dr.text((32, 26), "Medication round" if n > 1 else "Bedside delivery", font=self.font["title"], fill=INK)
        dr.text((32, 62), phase, font=self.font["body"], fill=INK)
        dr.text((32, 92), f"stop {min(k + 1, n)} of {n}   ·   t = {info['clock']:5.1f} s   ·   {self.speed:g}x speed",
                font=self.font["small"], fill=(90, 95, 110))
        #stop chips along the bottom left
        done = set(info["done"])
        x = 16
        for i, st in enumerate(info["stops"]):
            label = f"{i + 1}  {st['label']}"
            tw = dr.textlength(label, font=self.font["chip"]) + 26
            colour = (GREEN if st["name"] in done else ORANGE if i == k else (190, 190, 200)) + (235,)
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(overlay).rounded_rectangle((x, self.H - 50, x + tw, self.H - 18), 16, fill=colour)
            img.alpha_composite(overlay)
            dr = ImageDraw.Draw(img)
            dr.text((x + 13, self.H - 34), label, font=self.font["chip"], fill=WHITE, anchor="lm")
            x += tw + 8
        #robot camera, top right
        if inset_rgb is not None:
            ix = W - 336
            self.panel(img, (ix - 8, 16, W - 8, 290))
            dr = ImageDraw.Draw(img)
            dr.text((ix, 22), "Robot camera", font=self.font["small"], fill=INK)
            seen_now = [i for i, t in self.last_seen.items() if info["clock"] - t < 0.3]
            if seen_now:
                dr.text((W - 16, 22), "marker " + ", ".join(map(str, sorted(seen_now))) + " read",
                        font=self.font["small"], fill=GREEN, anchor="ra")
            img.paste(Image.fromarray(inset_rgb), (ix, 44))
        #map, bottom right
        mm = self.minimap(info)
        mx, my = W - 336, self.H - 50 - mm.size[1] - 8
        self.panel(img, (mx - 8, my - 8, W - 8, my + mm.size[1] + 8))
        img.paste(mm, (mx, my))
        return np.asarray(img.convert("RGB"))

    def summary_card(self, rgb, results, stop_log):
        img = Image.fromarray(rgb).convert("RGBA")
        rows = len(stop_log)
        title = "Round complete" if results["success"] else "Round finished with a problem"
        dr = ImageDraw.Draw(img)
        w = int(max(620, dr.textlength(title, font=self.font["card_title"]) + 64))
        h = 150 + 38 * rows
        x0, y0 = (self.W - w) // 2, (self.H - h) // 2
        self.panel(img, (x0, y0, x0 + w, y0 + h), radius=22, alpha=232)
        dr = ImageDraw.Draw(img)
        dr.text((x0 + 32, y0 + 24), title, font=self.font["card_title"], fill=INK)
        dr.text((x0 + 32, y0 + 72), f"{results['stops_reached']}/{results['stops_total']} stops   ·   "
                f"{results['time_s']:.0f} s   ·   {results['collision_ticks']} collisions",
                font=self.font["body"], fill=(90, 95, 110))
        y = y0 + 112
        for st in stop_log:
            ok = st["error_m"] < 0.05 and abs(st["heading_deg"]) < 5
            dr.ellipse([x0 + 32, y + 4, x0 + 50, y + 22], fill=GREEN if ok else ORANGE)
            dr.text((x0 + 64, y), st["label"], font=self.font["card"], fill=INK)
            dr.text((x0 + w - 32, y), f"{100 * st['error_m']:.2f} cm   {st['heading_deg']:+.1f}°",
                    font=self.font["card"], fill=INK, anchor="ra")
            y += 38
        return np.asarray(img.convert("RGB"))

    # -------------------------------------------------------------- rendering ---------
    def render_main(self, params, info=None):
        cam = self.camera(params)
        if self.sun >= 0 and self.shadows:
            self.m.light_castshadow[self.sun] = 1
        self.main.update_scene(self.d, cam)
        if info is not None:
            self.annotate(self.main.scene, info)
        rgb = self.main.render().copy()
        if self.sun >= 0:
            self.m.light_castshadow[self.sun] = 0
        return rgb

    def render_inset(self):
        if not self.has_robot_cam:
            return None
        self.inset.update_scene(self.d, camera=F.CAMERA_NAME)
        return self.inset.render().copy()

    def watch(self, info):
        #called by run_delivery after every 20 ms tick
        if self.m is None:
            self._setup(info["model"], info["data"])
        for marker_id in info["seen"]:
            self.last_seen[marker_id] = info["clock"]
        self.tick += 1
        self.last_info = info
        if self.tick % self.ticks_per_frame:
            return
        pose = self.true_pose()
        if not self.trail or np.hypot(*(pose[:2] - self.trail[-1])) > 0.03:
            self.trail.append(pose[:2].copy())
        target = self.follow_cam(pose)
        #ease from the whole-ward view into the follow camera over the first 5 s
        target = self.blend(self.overview_cam(), target, smoothstep(info["clock"] / 5.0))
        if self.cam_state is None:
            self.cam_state = target
        a = 1 - math.exp(-self.ticks_per_frame * 0.02 / 0.5)                #0.5 s smoothing
        self.cam_state = self.blend(self.cam_state, target, a)
        main = self.render_main(self.cam_state, info)
        frame = self.compose(main, info, self.render_inset())
        self.video.add(frame)
        self.frames += 1
        if self.frames % 120 == 0:
            print(f"  filmed {self.frames} frames (t = {info['clock']:.0f} s)", flush=True)

    def outro(self, results, run_log):
        #pull back to the whole ward, then hold on a summary card
        info = dict(self.last_info)
        info["phase"] = "stopped"
        info["done"] = [s["name"] for s in run_log["stops"]]
        start = dict(self.cam_state)
        for i in range(int(2.5 * self.fps)):
            params = self.blend(start, self.overview_cam(), smoothstep(i / (2.5 * self.fps)))
            self.video.add(self.compose(self.render_main(params, info), info, self.render_inset()))
        final = self.render_main(self.overview_cam(), info)
        card = self.summary_card(self.compose(final, info, self.render_inset()), results, run_log["stops"])
        for _ in range(int(4 * self.fps)):
            self.video.add(card)
        self.video.close()


# =====================================================================================#
# ------------------------------------- STILLS --------------------------------------- #
# =====================================================================================#

def stills(out_dir, width=1920, height=1080):
    m = mujoco.MjModel.from_xml_path(str(SCENE_OUT))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    m.vis.quality.offsamples = 8
    for i in range(m.nlight):
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, i) == "sun":
            m.light_castshadow[i] = 1
    r = mujoco.Renderer(m, height, width)
    saved = []
    for cam in ("overview", "corner", "top"):
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam) < 0:
            continue
        r.update_scene(d, camera=cam)
        path = out_dir / f"ward_{cam}.png"
        cv2.imwrite(str(path), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
        saved.append(path.name)
    r.close()
    print("saved", ", ".join(saved), "in", out_dir)


if __name__ == "__main__":
    from Analysis.conditions import CONDITIONS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true", help="only save pictures of the ward")
    ap.add_argument("--quick", action="store_true", help="faster preview: 960x540, no shadows, 4x speed")
    ap.add_argument("--condition", choices=sorted(CONDITIONS), default="nominal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=None, help="playback speed (default 3x, 4x with --quick)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", default=None, help="video file (default media/<ward>_round.mp4)")
    args = ap.parse_args()

    MEDIA.mkdir(exist_ok=True)
    if args.stills:
        stills(MEDIA)
        sys.exit(0)

    speed = args.speed or (4.0 if args.quick else 3.0)
    size = (960, 540) if args.quick else (1280, 720)
    tag = "" if args.condition == "nominal" else f"_{args.condition}"
    out = Path(args.out) if args.out else MEDIA / f"{WARD}_round{tag}.mp4"
    film = Film(out, *size, fps=args.fps, speed=speed, shadows=not args.quick)
    cond = TrialConditions(**{**CONDITIONS[args.condition], "name": args.condition, "seed": args.seed})
    t0 = time.time()
    results, run_log, *_ = run_delivery(cond, watch=film.watch)
    film.outro(results, run_log)
    print(f"Saved {out} ({film.video.kind}, {film.frames} frames + outro) in {time.time() - t0:.0f} s; "
          f"round {'succeeded' if results['success'] else 'FAILED'}")
