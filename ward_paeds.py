# =====================================================================================#
# ward_paeds.py - the paediatric ward: 4 beds, a marker at every bedside, a charging dock
# =====================================================================================#
#Build_scene.py loads this when the ward is "paeds" (the default). it holds:
#   1. the layout numbers, under the same names the classic ward uses, so every other
#      file (planner, LiDAR map, driver...) works without knowing which ward it is in
#   2. write_textures(): draws the floor, bed name plates, mural, windows and signs
#   3. scene_xml(): writes the decorated MuJoCo scene
#
#two kinds of geom:
#   SOLID (group 0)  walls, bed bases, cabinets, dock, counter, trolley, toy chest, planters.
#                    these collide, the LiDAR sees them, and they are in OBSTACLES for the planner
#   DECOR (group 2)  mattresses, blankets, teddies, curtains, lamps, murals, rugs...
#                    no collisions and the LiDAR skips group 2, so decoration can never change
#                    what the robot senses or where it can drive. only the camera sees it
#
#layout (metres, x = east, y = north, origin at the south-west floor corner):
#mission: pharmacy -> bed 1 -> bed 3 -> bed 4 -> bed 2 -> charging dock

import math
from pathlib import Path

import cv2
import numpy as np

#The layout

ROOM_X, ROOM_Y = 8.0, 6.0
WALL_THICKNESS = 0.10
WALL_HEIGHT = 1.65              #north, east and west walls
CUTAWAY_HEIGHT = 0.35           #south wall kept low so overhead cameras can see in (LiDAR still sees it)

#start on the pharmacy pad, facing east down the aisle
START_X, START_Y, START_THETA = 1.0, 3.0, 0.0
#random starts for repeated trials (Analysis/run_trials.py): around the pharmacy pad
START_REGION = ((0.85, 1.6), (2.4, 3.6))

MARKER_Z = 0.15 #marker centre height is same as the classic ward
BED_W, BED_L = 0.90, 1.90  #a child's bed: 0.9 m wide, 1.9 m long
BED_BASE_H = 0.40 #solid base (what the LiDAR sees)
CAB_W, CAB_D, CAB_H = 0.50, 0.55, 0.60 #bedside cabinet width, depth from the wall, height

#each bed: number, west edge, which wall its head is against, colour theme, name-plate icon
BEDS = [
    {"n": 1, "x0": 2.0, "side": "north", "theme": "sky", "teddy": "honey", "icon": "star"},
    {"n": 2, "x0": 5.0, "side": "north", "theme": "peach", "teddy": "cocoa", "icon": "sun"},
    {"n": 3, "x0": 2.0, "side": "south", "theme": "mint", "teddy": "rose", "icon": "heart"},
    {"n": 4, "x0": 5.0, "side": "south", "theme": "lavender", "teddy": "oat", "icon": "moon"},
]

#charging dock against the east wall, facing west down the aisle
DOCK_BOX = (7.55, 2.65, 8.0, 3.35, 0.42)
#where the trolley gets wheeled to in the moved_trolley test: across the aisle between bed 1 and bed 3
TROLLEY_MOVE_TO = (3.3, 3.0)
TROLLEY_HOME = (4.2, 0.25)
TROLLEY_HALF = (0.30, 0.20, 0.425)


def _bed_geometry(b):
    #head_y = the wall the head is against, d = which way the bed runs into the room
    north = b["side"] == "north"
    head_y = ROOM_Y if north else 0.0
    d = -1.0 if north else 1.0
    return head_y, d


def _cabinet_box(b):
    #the cabinet sits east of the bed head, against the same wall. the 0.25 m gap leaves the
    #robot's head-on approach line 0.50 m from the bed edge: the planner keeps 0.30 m, plus room
    #for the few centimetres a LiDAR-drawn bed edge can be smeared by
    head_y, d = _bed_geometry(b)
    x0 = b["x0"] + BED_W + 0.25
    ya, yb = head_y, head_y + d * CAB_D
    return x0, min(ya, yb), x0 + CAB_W, max(ya, yb)


def _markers():
    out = []
    for b in BEDS:
        head_y, d = _bed_geometry(b)
        cx0, cy0, cx1, cy1 = _cabinet_box(b)
        face_y = head_y + d * CAB_D
        #the marker picture sits 1 mm proud of the face; its position is where the picture IS
        out.append({"name": f"bed_{b['n']}", "id": b["n"], "x": (cx0 + cx1) / 2, "y": face_y + d * 0.001,
                    "z": MARKER_Z, "facing": math.pi / 2 if d > 0 else -math.pi / 2,
                    "label": f"Bed {b['n']}"})
    x0, y0, x1, y1, _ = DOCK_BOX
    out.append({"name": "dock", "id": 7, "x": x0 - 0.001, "y": (y0 + y1) / 2, "z": MARKER_Z,
                "facing": math.pi, "label": "Charging dock"})
    return out


#every marker the robot knows: name, ArUco id, face centre, height, direction the face points (rad)
MARKERS = _markers()
#the order of the round. the last stop is where the robot finishes (the charging dock)
MISSION = ["bed_1", "bed_3", "bed_4", "bed_2", "dock"]


def _obstacles():
    obs = []
    for b in BEDS:
        head_y, d = _bed_geometry(b)
        ya, yb = head_y, head_y + d * BED_L
        obs.append((f"bed_{b['n']}", b["x0"], min(ya, yb), b["x0"] + BED_W, max(ya, yb), BED_BASE_H))
        obs.append((f"cabinet_{b['n']}", *_cabinet_box(b), CAB_H))
    x0, y0, x1, y1, h = DOCK_BOX
    obs.append(("dock", x0, y0, x1, y1, h))
    obs.append(("pharmacy_counter", 0.0, 2.2, 0.45, 3.8, 0.90))
    tx, ty = TROLLEY_HOME
    obs.append(("trolley", tx - TROLLEY_HALF[0], ty - TROLLEY_HALF[1],
                tx + TROLLEY_HALF[0], ty + TROLLEY_HALF[1], 2 * TROLLEY_HALF[2]))
    obs.append(("toy_chest", 4.0, 5.55, 4.6, 6.0, 0.45))
    for name, x0, y0 in (("planter_sw", 0.0, 0.0), ("planter_nw", 0.0, 5.55),
                         ("planter_ne", 7.55, 5.55), ("planter_se", 7.55, 0.0)):
        obs.append((name, x0, y0, x0 + 0.45, y0 + 0.45, 0.40))
    return obs


#(name, x_min, y_min, x_max, y_max, height) of everything solid, same format as the classic ward
OBSTACLES = _obstacles()

#colouring

PALETTE = {
    "cream": (0.99, 0.96, 0.90), "white": (0.97, 0.97, 0.96), "snow": (1.0, 1.0, 1.0),
    "sky": (0.60, 0.79, 0.96), "peach": (1.00, 0.72, 0.60), "mint": (0.55, 0.86, 0.73),
    "lavender": (0.76, 0.68, 0.95), "butter": (1.00, 0.86, 0.45), "rose": (1.00, 0.68, 0.78),
    "coral": (0.98, 0.56, 0.56), "teal": (0.42, 0.76, 0.76), "leaf": (0.40, 0.70, 0.42),
    "leaf2": (0.52, 0.80, 0.46), "leaf3": (0.33, 0.62, 0.40), "honey": (0.85, 0.62, 0.35),
    "cocoa": (0.62, 0.44, 0.32), "oat": (0.92, 0.84, 0.70), "ink": (0.13, 0.13, 0.18),
    "steel": (0.62, 0.65, 0.70), "wood": (0.90, 0.78, 0.62), "lilac_ground": (0.86, 0.84, 0.90),
}


def mix(c, t, other=(1.0, 1.0, 1.0)):
    #blend colour c towards other (white by default) by t (0 = c, 1 = other)
    return tuple(a + (b - a) * t for a, b in zip(c, other))


def shade(c, t):
    return mix(c, t, (0.0, 0.0, 0.0))


def rgba(c, a=1.0):
    return f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {a:.2f}"


def _bgr(c):
    #matplotlib-style 0-1 RGB -> OpenCV 0-255 BGR
    return tuple(int(round(255 * v)) for v in c[::-1])

#textures

ASSET_DIR = "ward_assets"


def _rounded_rect(img, x0, y0, x1, y1, r, colour):
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), colour, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), colour, -1, cv2.LINE_AA)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(img, (cx, cy), r, colour, -1, cv2.LINE_AA)


def _centred_text(img, text, cx, cy, scale, colour, thickness):
    font = cv2.FONT_HERSHEY_DUPLEX
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (int(cx - w / 2), int(cy + h / 2)), font, scale, colour, thickness, cv2.LINE_AA)


def _star(img, cx, cy, r, colour):
    pts = [(cx + (r if k % 2 == 0 else 0.45 * r) * math.sin(k * math.pi / 5),
            cy - (r if k % 2 == 0 else 0.45 * r) * math.cos(k * math.pi / 5)) for k in range(10)]
    cv2.fillPoly(img, [np.array(pts, np.int32)], colour, cv2.LINE_AA)


def _heart(img, cx, cy, r, colour):
    cv2.circle(img, (int(cx - 0.5 * r), int(cy - 0.25 * r)), int(0.55 * r), colour, -1, cv2.LINE_AA)
    cv2.circle(img, (int(cx + 0.5 * r), int(cy - 0.25 * r)), int(0.55 * r), colour, -1, cv2.LINE_AA)
    pts = np.array([(cx - 1.02 * r, cy - 0.05 * r), (cx + 1.02 * r, cy - 0.05 * r), (cx, cy + 1.0 * r)], np.int32)
    cv2.fillPoly(img, [pts], colour, cv2.LINE_AA)


def _sun(img, cx, cy, r, colour):
    for k in range(12):
        a = k * math.pi / 6
        cv2.line(img, (int(cx + 1.15 * r * math.cos(a)), int(cy + 1.15 * r * math.sin(a))),
                 (int(cx + 1.55 * r * math.cos(a)), int(cy + 1.55 * r * math.sin(a))), colour, max(2, r // 7), cv2.LINE_AA)
    cv2.circle(img, (int(cx), int(cy)), int(r), colour, -1, cv2.LINE_AA)


def _moon(img, cx, cy, r, colour, background):
    cv2.circle(img, (int(cx), int(cy)), int(r), colour, -1, cv2.LINE_AA)
    cv2.circle(img, (int(cx + 0.45 * r), int(cy - 0.25 * r)), int(0.85 * r), background, -1, cv2.LINE_AA)


def _cloud(img, cx, cy, s, colour):
    for dx, dy, rr in ((-0.9, 0.1, 0.55), (-0.35, -0.25, 0.7), (0.35, -0.15, 0.62), (0.9, 0.12, 0.5), (0.0, 0.2, 0.6)):
        cv2.circle(img, (int(cx + dx * s), int(cy + dy * s)), int(rr * s), colour, -1, cv2.LINE_AA)


def _icon(img, kind, cx, cy, r, colour, background):
    if kind == "star":
        _star(img, cx, cy, r, colour)
    elif kind == "heart":
        _heart(img, cx, cy, r, colour)
    elif kind == "sun":
        _sun(img, cx, cy, int(0.62 * r), colour)
    else:
        _moon(img, cx, cy, r, colour, background)


def _floor_texture():
    #8 x 6 m at 200 px/m: soft cream tiles (0.5 m) with a sprinkle of pastel tiles and little stars
    px = 200
    h, w = int(ROOM_Y * px), int(ROOM_X * px)
    img = np.zeros((h, w, 3), np.uint8)
    tile = px // 2
    rng = np.random.default_rng(4)
    accents = ["sky", "peach", "mint", "lavender", "butter", "rose"]
    for r in range(h // tile):
        for c in range(w // tile):
            base = (0.985, 0.955, 0.905) if (r + c) % 2 == 0 else (0.965, 0.93, 0.875)
            if rng.random() < 0.09:
                base = mix(PALETTE[accents[rng.integers(len(accents))]], 0.45)
            y0, x0 = r * tile, c * tile
            img[y0:y0 + tile, x0:x0 + tile] = _bgr(base)
            if rng.random() < 0.05:
                _star(img, x0 + tile / 2, y0 + tile / 2, tile * 0.22, _bgr(mix(base, 0.55)))
    for k in range(0, h + 1, tile):
        cv2.line(img, (0, k), (w, k), _bgr((0.90, 0.86, 0.80)), 2)
    for k in range(0, w + 1, tile):
        cv2.line(img, (k, 0), (k, h), _bgr((0.90, 0.86, 0.80)), 2)
    return img


def _plaque_texture(b):
    w, h = 600, 240
    theme = PALETTE[b["theme"]]
    img = np.full((h, w, 3), 255, np.uint8)
    _rounded_rect(img, 0, 0, w - 1, h - 1, 50, _bgr(shade(theme, 0.10)))
    _rounded_rect(img, 14, 14, w - 15, h - 15, 40, _bgr(theme))
    _icon(img, b["icon"], 130, h / 2, 62, _bgr((1.0, 0.97, 0.80)), _bgr(theme))
    _centred_text(img, f"BED {b['n']}", 385, h / 2, 2.6, _bgr(PALETTE["snow"]), 6)
    return img


def _mural_texture():
    #a rainbow over little hills, with a sun and clouds
    w, h = 1260, 850
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):                                     #sky gradient
        t = y / h
        img[y, :] = _bgr(mix((0.70, 0.86, 1.0), 0.35 + 0.5 * t))
    cx, cy = w // 2, int(h * 0.98)
    for k, name in enumerate(["coral", "peach", "butter", "mint", "sky", "lavender"]):
        r = int(520 - k * 42)
        cv2.ellipse(img, (cx, cy), (r, r), 0, 180, 360, _bgr(PALETTE[name]), 42, cv2.LINE_AA)
    _sun(img, 170, 160, 70, _bgr(PALETTE["butter"]))
    _cloud(img, 1030, 170, 85, _bgr(PALETTE["snow"]))
    _cloud(img, 760, 260, 60, _bgr(PALETTE["snow"]))
    _cloud(img, 330, 380, 55, _bgr(PALETTE["snow"]))
    cv2.ellipse(img, (250, h + 140), (520, 330), 0, 0, 360, _bgr(PALETTE["leaf2"]), -1, cv2.LINE_AA)
    cv2.ellipse(img, (1000, h + 170), (560, 330), 0, 0, 360, _bgr(mix(PALETTE["leaf"], 0.15)), -1, cv2.LINE_AA)
    for x, y, c in ((120, 760, "rose"), (210, 790, "butter"), (1080, 770, "lavender"), (1160, 800, "rose")):
        for k in range(5):
            a = k * 2 * math.pi / 5
            cv2.circle(img, (int(x + 14 * math.cos(a)), int(y + 14 * math.sin(a))), 11, _bgr(PALETTE[c]), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 8, _bgr(PALETTE["butter"]), -1, cv2.LINE_AA)
    _rounded_rect(img, 0, 0, w - 1, 22, 10, _bgr(PALETTE["snow"]))
    return img


def _window_texture():
    w, h = 1000, 600
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        img[y, :] = _bgr(mix((0.55, 0.78, 0.98), 0.15 + 0.55 * y / h))
    _cloud(img, 280, 200, 70, _bgr(PALETTE["snow"]))
    _cloud(img, 720, 330, 55, _bgr(PALETTE["snow"]))
    frame = _bgr(PALETTE["snow"])
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), frame, 40)
    cv2.line(img, (w // 2, 0), (w // 2, h), frame, 24)
    cv2.line(img, (0, h // 2), (w, h // 2), frame, 24)
    return img


def _sign_texture(text, colour, icon):
    w, h = 900, 180
    img = np.full((h, w, 3), 255, np.uint8)
    _rounded_rect(img, 0, 0, w - 1, h - 1, 40, _bgr(shade(colour, 0.12)))
    _rounded_rect(img, 10, 10, w - 11, h - 11, 32, _bgr(colour))
    white = _bgr(PALETTE["snow"])
    if icon == "cross":
        cv2.rectangle(img, (55, 70), (145, 110), white, -1)
        cv2.rectangle(img, (80, 45), (120, 135), white, -1)
    else:                                                                 #lightning bolt
        pts = np.array([(115, 30), (60, 100), (95, 100), (75, 152), (140, 75), (102, 75), (125, 30)], np.int32)
        cv2.fillPoly(img, [pts], white, cv2.LINE_AA)
    _centred_text(img, text, 520, h / 2, 2.0, white, 5)
    return img


def write_textures(folder):
    #draw every picture the scene uses into folder/ward_assets. returns nothing; the XML refers to the files
    out = Path(folder) / ASSET_DIR
    out.mkdir(exist_ok=True)
    #pictures are drawn upright: MuJoCo shows the first image row at the top of a vertical plane
    def save(name, img):
        cv2.imwrite(str(out / name), img)
    save("floor.png", _floor_texture())
    for b in BEDS:
        save(f"plaque_{b['n']}.png", _plaque_texture(b))
    save("mural.png", _mural_texture())
    save("window.png", _window_texture())
    save("sign_pharmacy.png", _sign_texture("PHARMACY", PALETTE["rose"], "cross"))
    save("sign_dock.png", _sign_texture("CHARGING DOCK", PALETTE["teal"], "bolt"))

#geometry stuff

def _v(*xs):
    return " ".join(f"{x:.4f}".rstrip("0").rstrip(".") if abs(x) > 1e-9 else "0" for x in xs)


def solid(name, x0, y0, x1, y1, z0, z1, colour):
    #a collision box: the physics, the LiDAR and the planner all see it
    return (f'<geom name="{name}" type="box" pos="{_v((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)}" '
            f'size="{_v((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2)}" rgba="{rgba(colour)}" '
            f'group="0" contype="3" conaffinity="3" material="matte"/>')


DECOR = 'contype="0" conaffinity="0" group="2" material="matte"'


def dbox(x0, y0, x1, y1, z0, z1, colour, a=1.0, extra=""):
    #decor box from its corner coordinates
    return (f'<geom type="box" pos="{_v((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)}" '
            f'size="{_v(abs(x1 - x0) / 2, abs(y1 - y0) / 2, abs(z1 - z0) / 2)}" rgba="{rgba(colour, a)}" {DECOR} {extra}/>')


def dgeom(kind, pos, size, colour, a=1.0, extra=""):
    #extra may name its own material (the glowing lamp shades), which then replaces "matte"
    decor = DECOR.replace(' material="matte"', "") if "material=" in extra else DECOR
    return f'<geom type="{kind}" pos="{_v(*pos)}" size="{_v(*size)}" rgba="{rgba(colour, a)}" {decor} {extra}/>'


def dcapsule(p, q, r, colour, kind="capsule", a=1.0):
    return f'<geom type="{kind}" fromto="{_v(*p, *q)}" size="{_v(r)}" rgba="{rgba(colour, a)}" {DECOR}/>'


def dplane(x, y, z, facing, half_w, half_h, material, name=None, proud=0.002):
    #a flat picture on a vertical surface, its front facing the angle 'facing' (rad),
    #'proud' metres in front of the point (x, y) so it never flickers into the surface
    nx, ny = math.cos(facing), math.sin(facing)
    tag = f'name="{name}" ' if name else ""
    return (f'<geom {tag}type="plane" pos="{_v(x + proud * nx, y + proud * ny, z)}" '
            f'xyaxes="{_v(-math.sin(facing), math.cos(facing), 0, 0, 0, 1)}" '
            f'size="{_v(half_w, half_h, 0.01)}" material="{material}" contype="0" conaffinity="0" group="2"/>')


def decal(x, y, r, colour, a=1.0, z=0.0015):
    #a flat coloured circle on the floor (rugs, stepping stones)
    return f'<geom type="cylinder" pos="{_v(x, y, z)}" size="{_v(r, 0.0012)}" rgba="{rgba(colour, a)}" {DECOR}/>'

# =====================================================================================#
# --------------------------------- FURNITURE ---------------------------------------- #
# =====================================================================================#

def bed_xml(b):
    theme = PALETTE[b["theme"]]
    head_y, d = _bed_geometry(b)
    x0, x1 = b["x0"], b["x0"] + BED_W
    cx = (x0 + x1) / 2

    def Y(s):                                   #distance s from the head -> world y
        return head_y + d * s

    def span(xa, xb, sa, sb, za, zb, colour, a=1.0):
        return dbox(xa, Y(sa), xb, Y(sb), za, zb, colour, a)

    g = [f"<!-- bed {b['n']} -->"]
    ya, yb = Y(0), Y(BED_L)
    g.append(solid(f"bed_{b['n']}", x0, min(ya, yb), x1, max(ya, yb), 0.0, BED_BASE_H, mix(theme, 0.80)))
    g.append(span(x0 - 0.004, x1 + 0.004, 0.08, BED_L - 0.02, 0.05, 0.13, theme))          #coloured skirt band
    g.append(span(x0 - 0.02, x1 + 0.02, 0.0, 0.06, 0.0, 0.95, theme))                      #headboard
    g.append(dcapsule((x0 - 0.02, Y(0.03), 0.95), (x1 + 0.02, Y(0.03), 0.95), 0.04, shade(theme, 0.05)))
    g.append(span(x0 - 0.01, x1 + 0.01, BED_L - 0.05, BED_L, 0.0, 0.66, theme))           #footboard
    g.append(dcapsule((x0 - 0.01, Y(BED_L - 0.025), 0.66), (x1 + 0.01, Y(BED_L - 0.025), 0.66), 0.03, shade(theme, 0.05)))
    g.append(span(x0 + 0.02, x1 - 0.02, 0.06, BED_L - 0.05, BED_BASE_H, 0.54, PALETTE["snow"]))   #mattress
    #pillow, then the duvet with a turned-down sheet and polka dots
    g.append(dgeom("ellipsoid", (cx, Y(0.26), 0.585), (0.33, 0.14, 0.06), (1.0, 0.99, 0.97)))
    g.append(span(x0 - 0.015, x1 + 0.015, 0.62, BED_L - 0.06, 0.30, 0.565, mix(theme, 0.25)))
    g.append(span(x0 - 0.018, x1 + 0.018, 0.58, 0.68, 0.30, 0.572, PALETTE["snow"]))
    for i, s in enumerate(np.arange(0.86, BED_L - 0.15, 0.2)):
        for u in ((0.2, 0.5, 0.8) if i % 2 == 0 else (0.35, 0.65)):
            g.append(dgeom("cylinder", (x0 + u * BED_W, Y(s), 0.567), (0.032, 0.002), PALETTE["snow"]))
    #cot rails on both sides
    for xr in (x0 + 0.015, x1 - 0.015):
        g.append(dcapsule((xr, Y(0.35), 0.76), (xr, Y(1.45), 0.76), 0.014, PALETTE["snow"]))
        for s in np.linspace(0.35, 1.45, 6):
            g.append(dcapsule((xr, Y(s), 0.55), (xr, Y(s), 0.76), 0.009, PALETTE["snow"]))
    #a teddy sitting by the pillow
    t = PALETTE[b["teddy"]]
    tx, ts = cx + 0.2, 0.42
    f = d                                         #teddy faces down the bed
    g += [dgeom("ellipsoid", (tx, Y(ts), 0.63), (0.07, 0.06, 0.085), t),
          dgeom("ellipsoid", (tx, Y(ts) + f * 0.035, 0.62), (0.045, 0.03, 0.055), mix(t, 0.4)),
          dgeom("sphere", (tx, Y(ts), 0.75), (0.058,), t),
          dgeom("sphere", (tx - 0.042, Y(ts), 0.795), (0.022,), t),
          dgeom("sphere", (tx + 0.042, Y(ts), 0.795), (0.022,), t),
          dgeom("sphere", (tx, Y(ts) + f * 0.048, 0.738), (0.024,), mix(t, 0.45)),
          dgeom("sphere", (tx, Y(ts) + f * 0.07, 0.745), (0.008,), PALETTE["ink"]),
          dgeom("sphere", (tx - 0.022, Y(ts) + f * 0.05, 0.77), (0.008,), PALETTE["ink"]),
          dgeom("sphere", (tx + 0.022, Y(ts) + f * 0.05, 0.77), (0.008,), PALETTE["ink"]),
          dgeom("ellipsoid", (tx - 0.07, Y(ts) + f * 0.02, 0.64), (0.025, 0.025, 0.045), t),
          dgeom("ellipsoid", (tx + 0.07, Y(ts) + f * 0.02, 0.64), (0.025, 0.025, 0.045), t),
          dgeom("sphere", (tx - 0.04, Y(ts) + f * 0.07, 0.565), (0.03,), t),
          dgeom("sphere", (tx + 0.04, Y(ts) + f * 0.07, 0.565), (0.03,), t)]
    #name plate on the headboard, facing down the bed
    g.append(dplane(cx, Y(0.06), 0.80, math.pi / 2 if d > 0 else -math.pi / 2, 0.24, 0.096, f"plaque_{b['n']}"))
    #privacy curtain gathered against the wall on the bed's west side, on a ceiling track.
    #north beds only: on the south beds it would block the overhead cameras
    if b["side"] == "north":
        xc = x0 - 0.07
        g.append(dcapsule((xc, Y(0.0), 1.80), (xc, Y(1.1), 1.80), 0.010, PALETTE["steel"]))
        for i in range(7):
            s = 0.02 + i * 0.075
            off = 0.012 if i % 2 else -0.012
            g.append(dbox(xc + off - 0.004, Y(s), xc + off + 0.004, Y(s + 0.08), 0.55, 1.78, mix(theme, 0.30), a=0.92))
    #round rug at the foot of the bed
    g.append(decal(cx, Y(BED_L + 0.45), 0.42, theme))
    g.append(decal(cx, Y(BED_L + 0.45), 0.33, mix(theme, 0.45), z=0.0025))
    return g


def cabinet_xml(b):
    theme = PALETTE[b["theme"]]
    head_y, d = _bed_geometry(b)
    x0, y0, x1, y1 = _cabinet_box(b)
    cx = (x0 + x1) / 2
    face = head_y + d * CAB_D
    g = [f"<!-- cabinet {b['n']} -->", solid(f"cabinet_{b['n']}", x0, y0, x1, y1, 0.0, CAB_H, PALETTE["white"])]
    g.append(dbox(x0 - 0.01, y0 - 0.01, x1 + 0.01, y1 + 0.01, CAB_H, CAB_H + 0.025, mix(theme, 0.2)))     #top
    for za, zb in ((0.30, 0.43), (0.45, 0.58)):                                                   #drawers
        g.append(dbox(x0 + 0.035, face, x1 - 0.035, face + d * 0.012, za, zb, mix(theme, 0.15)))
        g.append(dgeom("sphere", (cx, face + d * 0.02, (za + zb) / 2), (0.018,), PALETTE["snow"]))
    #a little lamp and a medicine cup on top
    top = CAB_H + 0.025
    lx, ly = cx - 0.1, (y0 + y1) / 2 - d * 0.08
    g += [dgeom("cylinder", (lx, ly, top + 0.012), (0.055, 0.012), PALETTE["snow"]),
          dcapsule((lx, ly, top + 0.02), (lx, ly, top + 0.2), 0.008, PALETTE["steel"]),
          dgeom("cylinder", (lx, ly, top + 0.24), (0.085, 0.055), mix(PALETTE["butter"], 0.3), extra='material="glow_soft"'),
          dgeom("cylinder", (cx + 0.12, ly + d * 0.12, top + 0.03), (0.03, 0.03), mix(PALETTE["rose"], 0.2)),
          dgeom("cylinder", (cx + 0.12, ly + d * 0.12, top + 0.062), (0.026, 0.003), PALETTE["snow"])]
    return g


def dock_xml():
    x0, y0, x1, y1, h = DOCK_BOX
    cy = (y0 + y1) / 2
    teal = PALETTE["teal"]
    g = ["<!-- charging dock -->", solid("dock", x0, y0, x1, y1, 0.0, h, mix(PALETTE["butter"], 0.15))]
    g.append(dcapsule((x0 + 0.06, y0, h), (x0 + 0.06, y1, h), 0.06, mix(PALETTE["butter"], 0.05)))
    g.append(dbox(x0 - 0.004, y0 + 0.05, x0, y1 - 0.05, 0.27, 0.40, teal))                      #face panel
    g.append(dbox(x0 - 0.006, cy - 0.12, x0, cy - 0.07, 0.015, 0.045, PALETTE["steel"]))      #charging contacts
    g.append(dbox(x0 - 0.006, cy + 0.07, x0, cy + 0.12, 0.015, 0.045, PALETTE["steel"]))
    g.append(dgeom("sphere", ((x0 + x1) / 2, cy, h + 0.06), (0.05,), (0.45, 1.0, 0.6), extra='material="glow_bright"'))
    #parking bay on the floor in front of it
    px = x0 - 0.40
    g.append(dbox(px - 0.36, cy - 0.36, px + 0.36, cy + 0.36, 0.0005, 0.0025, mix(teal, 0.25)))
    g.append(dbox(px - 0.30, cy - 0.30, px + 0.30, cy + 0.30, 0.0006, 0.0028, mix(teal, 0.75)))
    g.append(dplane(ROOM_X, cy, 1.05, math.pi, 0.45, 0.09, "sign_dock"))
    return g


def pharmacy_xml():
    rose = PALETTE["rose"]
    g = ["<!-- pharmacy -->", solid("pharmacy_counter", 0.0, 2.2, 0.45, 3.8, 0.0, 0.90, PALETTE["white"])]
    g.append(dbox(0.0, 2.17, 0.49, 3.83, 0.90, 0.94, PALETTE["wood"]))                           #worktop
    g.append(dbox(0.45, 2.25, 0.455, 3.75, 0.55, 0.72, rose))                                    #front stripe
    g.append(dbox(0.45, 2.25, 0.455, 3.75, 0.08, 0.14, mix(rose, 0.4)))
    #shelves of medicine on the wall above
    colours = ["sky", "peach", "mint", "lavender", "butter", "rose", "teal", "coral"]
    for k, z in enumerate((1.10, 1.32)):
        g.append(dbox(0.0, 2.3, 0.22, 3.7, z - 0.02, z, PALETTE["wood"]))
        for i, y in enumerate(np.arange(2.38, 3.64, 0.105)):
            c = PALETTE[colours[(i + 3 * k) % len(colours)]]
            if (i + k) % 3 == 0:
                g.append(dgeom("cylinder", (0.11, y, z + 0.06), (0.035, 0.06), mix(c, 0.1)))
                g.append(dgeom("cylinder", (0.11, y, z + 0.13), (0.02, 0.012), PALETTE["snow"]))
            else:
                g.append(dbox(0.06, y - 0.04, 0.17, y + 0.04, z, z + 0.09 + 0.03 * ((i + k) % 2), c))
    #things on the counter: a basket of medicine boxes ready for the round
    g.append(dbox(0.12, 2.6, 0.38, 3.0, 0.94, 1.02, mix(PALETTE["wood"], 0.0, (0.75, 0.6, 0.45))))
    for i, y in enumerate((2.66, 2.76, 2.86, 2.94)):
        g.append(dbox(0.17, y - 0.035, 0.33, y + 0.035, 1.0, 1.06, PALETTE[colours[i]]))
    g.append(dplane(0.0, 3.0, 1.52, 0.0, 0.5, 0.1, "sign_pharmacy"))
    #start pad: a blue circle with a white cross
    g.append(decal(START_X, START_Y, 0.30, mix(PALETTE["sky"], 0.1)))
    g.append(dbox(START_X - 0.15, START_Y - 0.04, START_X + 0.15, START_Y + 0.04, 0.0006, 0.003, PALETTE["snow"]))
    g.append(dbox(START_X - 0.04, START_Y - 0.15, START_X + 0.04, START_Y + 0.15, 0.0006, 0.003, PALETTE["snow"]))
    return g


def trolley_xml():
    tx, ty = TROLLEY_HOME
    hx, hy, hz = TROLLEY_HALF
    mint = PALETTE["mint"]
    #the trolley is its own body so the moved_trolley test can wheel the whole thing away
    g = [f'<body name="trolley_body" pos="{_v(tx, ty, 0)}">',
         f'  <geom name="trolley" type="box" pos="0 0 {_v(hz)}" size="{_v(hx, hy, hz)}" '
         f'rgba="{rgba(PALETTE["white"])}" group="0" contype="3" conaffinity="3"/>']
    for k, (za, zb) in enumerate(((0.18, 0.36), (0.40, 0.58), (0.62, 0.78))):
        g.append("  " + dbox(-hx + 0.03, -hy - 0.006, hx - 0.03, -hy, za, zb, mix(mint, 0.15 * k)))
        g.append("  " + dbox(-hx + 0.03, hy, hx - 0.03, hy + 0.006, za, zb, mix(mint, 0.15 * k)))
    g.append("  " + dcapsule((-hx - 0.05, -hy + 0.04, 0.80), (-hx - 0.05, hy - 0.04, 0.80), 0.015, PALETTE["steel"]))
    g.append("  " + dbox(-hx - 0.01, -hy - 0.01, hx + 0.01, hy + 0.01, 2 * hz, 2 * hz + 0.02, mix(mint, 0.5)))
    for sx in (-1, 1):
        for sy in (-1, 1):
            g.append("  " + dgeom("sphere", (sx * (hx - 0.05), sy * (hy - 0.04), 0.03), (0.03,), PALETTE["ink"]))
    g.append("  " + dgeom("cylinder", (0.12, 0.0, 2 * hz + 0.07), (0.05, 0.05), mix(PALETTE["sky"], 0.2)))
    g.append("  " + dbox(-0.2, -0.1, 0.0, 0.1, 2 * hz + 0.02, 2 * hz + 0.06, PALETTE["peach"]))
    g.append("</body>")
    return g


def toys_and_plants_xml():
    g = ["<!-- toy chest -->", solid("toy_chest", 4.0, 5.55, 4.6, 6.0, 0.0, 0.45, mix(PALETTE["coral"], 0.25))]
    g.append(dbox(3.98, 5.53, 4.62, 6.0, 0.45, 0.49, mix(PALETTE["coral"], 0.05)))
    g.append(dgeom("sphere", (4.3, 5.548, 0.25), (0.06,), mix(PALETTE["butter"], 0.0)))
    g.append(dgeom("sphere", (4.15, 5.75, 0.56), (0.07,), PALETTE["sky"]))
    for i, (x, c) in enumerate(((4.33, "mint"), (4.43, "lavender"), (4.53, "peach"))):
        g.append(dbox(x - 0.04, 5.66, x + 0.04, 5.74, 0.49, 0.57, PALETTE[c]))
    g.append(dbox(4.39, 5.80, 4.47, 5.88, 0.57, 0.65, PALETTE["butter"]))
    #square planters with leafy plants in each corner
    pots = {"planter_sw": "lavender", "planter_nw": "peach", "planter_ne": "sky", "planter_se": "mint"}
    for name, x0, y0, x1, y1, h in OBSTACLES:
        if name not in pots:
            continue
        c = PALETTE[pots[name]]
        g.append(solid(name, x0, y0, x1, y1, 0.0, h, c))
        g.append(dbox(x0 - 0.01, y0 - 0.01, x1 + 0.01, y1 + 0.01, h - 0.04, h, shade(c, 0.08)))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rng = np.random.default_rng(len(name) + int(10 * x0))
        for k in range(9):
            a, r = k * 2.4, 0.06 + 0.012 * k
            z = h + 0.08 + 0.055 * k
            leaf = PALETTE[("leaf", "leaf2", "leaf3")[k % 3]]
            g.append(dgeom("sphere", (cx + r * math.cos(a), cy + r * math.sin(a), z), (0.12 - 0.006 * k,), leaf))
        for k in range(4):
            a = k * 1.7 + 0.4
            g.append(dgeom("sphere", (cx + 0.13 * math.cos(a), cy + 0.13 * math.sin(a), h + 0.25 + 0.1 * k),
                           (0.028,), PALETTE["rose" if k % 2 else "butter"]))
    #stepping stones down the aisle, in rainbow order
    colours = ["coral", "peach", "butter", "mint", "sky", "lavender", "rose"]
    for i, x in enumerate(np.arange(1.6, 7.0, 0.45)):
        if 2.95 < x < 3.55 or 5.95 < x < 6.55:          #keep the bed crossings clear
            continue
        g.append(decal(x, 3.0 + (0.12 if i % 2 else -0.12), 0.11, PALETTE[colours[i % 7]]))
    return g


def walls_xml():
    t, H = WALL_THICKNESS, WALL_HEIGHT
    upper, lower = (0.995, 0.975, 0.94), mix(PALETTE["mint"], 0.35)
    g = ["<!-- walls -->",
         solid("wall_S", -t, -t, ROOM_X + t, 0.0, 0.0, CUTAWAY_HEIGHT, lower),
         solid("wall_N", -t, ROOM_Y, ROOM_X + t, ROOM_Y + t, 0.0, H, upper),
         solid("wall_W", -t, 0.0, 0.0, ROOM_Y, 0.0, H, upper),
         solid("wall_E", ROOM_X, 0.0, ROOM_X + t, ROOM_Y, 0.0, H, upper)]
    #two-tone walls: mint below a white dado rail, cream above, white skirting
    rail = PALETTE["snow"]
    g += [dbox(0.0, ROOM_Y - 0.004, ROOM_X, ROOM_Y, 0.0, 0.62, lower), dbox(0.0, ROOM_Y - 0.012, ROOM_X, ROOM_Y, 0.62, 0.66, rail),
          dbox(0.0, 0.0, 0.004, ROOM_Y, 0.0, 0.62, lower), dbox(0.0, 0.0, 0.012, ROOM_Y, 0.62, 0.66, rail),
          dbox(ROOM_X - 0.004, 0.0, ROOM_X, ROOM_Y, 0.0, 0.62, lower), dbox(ROOM_X - 0.012, 0.0, ROOM_X, ROOM_Y, 0.62, 0.66, rail),
          dbox(-t, -t, ROOM_X + t, 0.0, CUTAWAY_HEIGHT, CUTAWAY_HEIGHT + 0.03, rail)]
    for y0, y1 in ((-t, ROOM_Y + t),):
        g.append(dbox(-t, y0, 0.0, y1, H, H + 0.03, rail))
        g.append(dbox(ROOM_X, y0, ROOM_X + t, y1, H, H + 0.03, rail))
    g.append(dbox(-t, ROOM_Y, ROOM_X + t, ROOM_Y + t, H, H + 0.03, rail))
    #pictures: rainbow mural between bed 1 and bed 2, windows on the east wall
    g.append(dplane(4.25, ROOM_Y, 1.05, -math.pi / 2, 0.62, 0.42, "mural"))
    g.append(dplane(ROOM_X, 1.35, 1.05, math.pi, 0.5, 0.3, "window"))
    g.append(dplane(ROOM_X, 4.65, 1.05, math.pi, 0.5, 0.3, "window"))
    return g


def marker_xml(m, marker_total):
    #ArUco marker: a flat picture on the cabinet (or dock) face, 1 mm proud of it
    return dplane(m["x"], m["y"], m["z"], m["facing"], marker_total / 2, marker_total / 2,
                  f"aruco_{m['id']}", name=f"aruco_marker_{m['id']}", proud=0.0)


def _lookat_xyaxes(pos, target):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return _v(*x, *y)


def scene_xml(robot_file, marker_total):
    #the whole ward as MJCF text. robot_file = the robot XML (with camera and LiDAR) to include
    textures = [f'<texture type="2d" name="aruco_{m["id"]}" file="marker_assets/aruco_{m["id"]}.png"/>'
                f'<material name="aruco_{m["id"]}" texture="aruco_{m["id"]}"/>' for m in MARKERS]
    textures.append('<material name="matte" specular="0.06" shininess="0.08"/>'
                    '<material name="glow_soft" emission="0.6" specular="0"/><material name="glow_bright" emission="1" specular="0"/>')
    pics = ["floor", "mural", "window", "sign_pharmacy", "sign_dock"] + [f"plaque_{b['n']}" for b in BEDS]
    for p in pics:
        textures.append(f'<texture type="2d" name="{p}" file="{ASSET_DIR}/{p}.png"/>'
                        f'<material name="{p}" texture="{p}" specular="0.03" shininess="0.05"/>')

    body = []
    body += walls_xml()
    for b in BEDS:
        body += bed_xml(b)
        body += cabinet_xml(b)
    body += dock_xml() + pharmacy_xml() + toys_and_plants_xml() + trolley_xml()
    body.append("<!-- ArUco markers: one per bedside cabinet, one on the dock -->")
    body += [marker_xml(m, marker_total) for m in MARKERS]

    #castshadow is off: shadows triple the robot camera's render time. make_video.py turns the
    #sun's shadow on for its own frames only
    lights = ['<light name="sun" directional="true" pos="2 -4 8" dir="0.25 0.55 -0.8" diffuse="0.44 0.42 0.40" '
              'specular="0 0 0" castshadow="false"/>',
              '<light name="fill" directional="true" pos="8 8 6" dir="-0.55 -0.35 -0.75" diffuse="0.26 0.26 0.30" '
              'specular="0 0 0" castshadow="false"/>']
    for b in BEDS:
        head_y, d = _bed_geometry(b)
        lights.append(f'<light pos="{_v(b["x0"] + BED_W / 2, head_y + d * 1.0, 2.4)}" dir="0 0 -1" '
                      f'diffuse="0.12 0.11 0.09" specular="0 0 0" cutoff="55" exponent="4" castshadow="false"/>')

    cams = [f'<camera name="overview" pos="4.0 -3.4 5.4" xyaxes="{_lookat_xyaxes((4.0, -3.4, 5.4), (4.0, 2.6, 0.0))}" fovy="50"/>',
            f'<camera name="corner" pos="-1.2 -1.6 3.6" xyaxes="{_lookat_xyaxes((-1.2, -1.6, 3.6), (3.6, 3.2, 0.2))}" fovy="55"/>',
            '<camera name="top" pos="4.0 3.0 9.5" xyaxes="1 0 0 0 1 0" fovy="40"/>']

    sx, sy, st = START_X, START_Y, START_THETA
    nl = "\n    "
    return f"""<mujoco model="paediatric ward">
  <!-- Robot: the ROBOTIS Waffle Pi with our camera and LiDAR site [ROB] -->
  <include file="{robot_file}"/>

  <statistic center="4 3 0.4" extent="5.5"/>
  <visual>
    <headlight diffuse="0.12 0.12 0.12" ambient="0.30 0.30 0.30" specular="0 0 0"/>
    <global offwidth="1920" offheight="1080" azimuth="90" elevation="-52"/>
    <quality shadowsize="4096" offsamples="0"/>
    <map znear="0.01" zfar="60"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.86 0.91 1.0" rgb2="0.99 0.95 0.92" width="512" height="3072"/>
    {nl.join(textures)}
  </asset>

  <worldbody>
    {nl.join(lights)}
    {nl.join(cams)}
    <!-- floor: the tiled picture covers the room exactly; it collides as an infinite plane -->
    <geom name="floor" type="plane" pos="{_v(ROOM_X / 2, ROOM_Y / 2, 0)}" size="{_v(ROOM_X / 2, ROOM_Y / 2, 0.05)}" material="floor"/>
    <geom type="plane" pos="{_v(ROOM_X / 2, ROOM_Y / 2, -0.003)}" size="9 8 0.05" rgba="{rgba(PALETTE['lilac_ground'])}" {DECOR}/>
    {nl.join(body)}
  </worldbody>

  <!-- "home" keyframe: robot x y z, orientation quaternion, then the two wheel angles -->
  <keyframe>
    <key name="home" qpos="{_v(sx, sy, 0)} {_v(math.cos(st / 2), 0, 0, math.sin(st / 2))} 0 0" ctrl="0 0"/>
  </keyframe>
</mujoco>
"""
