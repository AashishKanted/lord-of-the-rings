"""Generate the dimensioned top view directly from customer_cell.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT / "spec/customer_cell.json").read_text())
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="760" viewBox="0 0 900 760">',
       '<rect width="900" height="760" fill="white"/>',
       '<style>text{font-family:Arial,sans-serif;fill:#172b4d} .small{font-size:13px} .label{font-size:15px;font-weight:600} .dim{stroke:#8090a0;stroke-width:1;fill:none}</style>']


def text(x, y, value, kind="small", anchor="start"):
    svg.append(f'<text x="{x}" y="{y}" class="{kind}" text-anchor="{anchor}">{value}</text>')


def line(x1, y1, x2, y2, style='class="dim"'):
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" {style}/>')


def xy(x, y):
    return 50 + x*90, 610 - y*90


text(50, 30, "Advaitic | Synthetic conveyor assembly cell", "label")
text(50, 51, "Top view · dimensions in metres · z is vertical · not customer CAD")
svg.append('<rect x="50" y="70" width="540" height="540" fill="#f5f8fb" stroke="#172b4d" stroke-width="2"/>')
for value in range(1, 6):
    x, y = xy(value, value)
    line(x, 70, x, 610, 'stroke="#e2e8ef"')
    line(50, y, 590, y, 'stroke="#e2e8ef"')
    text(x, 632, str(value), anchor="middle")
text(50, 632, "0", anchor="middle")
text(590, 632, "6", anchor="middle")
text(320, 653, "World x (m)", anchor="middle")
for name, station in spec["stations"].items():
    cx, cy, _ = station["conveyor_center_world"]
    x, y = xy(cx, cy)
    svg.append(f'<rect x="{x-90}" y="{y-22.5}" width="180" height="45" rx="3" fill="#dce9f7" stroke="#1971c2"/>')
    direction = 1 if name == "A" else -1
    line(x-direction*50, y, x+direction*50, y, 'stroke="#1971c2" stroke-width="2"')
    tip = x + direction*50
    svg.append(f'<polygon points="{tip},{y} {tip-direction*9},{y-5} {tip-direction*9},{y+5}" fill="#1971c2"/>')
    text(x, y-34, f"{name}: 2.00 × 0.50 m, top z=0.75", anchor="middle")
    dx, dy, _ = station["dock_world"]
    px, py = xy(dx, dy)
    svg.append(f'<circle cx="{px}" cy="{py}" r="20" fill="#d3f9d8" stroke="#2b8a3e"/>')
    text(px+30, py+5, f"Dock {name} ({dx:.1f}, {dy:.1f})")
    # Entry/exit work interval and downstream inspection in station coordinates.
    for local_x, label in ((-.3, "entry"), (.3, "exit"), (.55, "inspect")):
        mx, _ = xy(cx+direction*local_x, cy)
        line(mx, y-22.5, mx, y+22.5, 'stroke="#e67700" stroke-dasharray="3 2"')
        text(mx, y+(57 if label == "inspect" else 39), label, anchor="middle")
sx, sy = xy(*spec["start_pose_world"][:2])
svg.append(f'<circle cx="{sx}" cy="{sy}" r="10" fill="#9775fa"/>')
text(sx-30, sy-20, "Start (1.0, 3.0)")
line(320, 410, 320, 270, 'stroke="#2b8a3e" stroke-width="2" stroke-dasharray="5 4"')
text(330, 345, "2.00 m dock spacing")
line(50, 678, 590, 678)
line(50, 670, 50, 686)
line(590, 670, 590, 686)
text(320, 701, "6.00 m", anchor="middle")
line(610, 70, 610, 610)
line(602, 70, 618, 70)
line(602, 610, 618, 610)
text(620, 340, "6.00 m")
text(685, 65, "Parts not to scale")
text(685, 90, "Rigid ring", "label")
svg.append('<circle cx="745" cy="155" r="48" fill="#74c0fc" stroke="#1971c2"/><circle cx="745" cy="155" r="28.8" fill="white" stroke="#1971c2"/>')
text(680, 225, "OD 80 / ID 48 mm")
text(680, 246, "Thickness 8 mm")
text(680, 295, "Bottle", "label")
for i, label in enumerate(["Height 180 mm", "Body Ø60 mm", "Neck Ø32 mm", "Heights from bottle base:", "Shoulder 120–140 mm", "Ring centre 130–136 mm"]):
    text(680, 320+i*23, label)
text(680, 470, "Operating targets", "label")
for i, label in enumerate(["Belt: 30 mm/s nominal", "Pitch: 450 mm / 15 s", "Assembly: ≤12 s", "Window: 600 mm / 20 s", "5 accepted at A, then B", "Mission: ≤270 s nominal"]):
    text(680, 495+i*23, label)
text(50, 738, "Station local +x follows belt travel. All bottle/ring offsets rotate with the kit at B.")
svg.append('</svg>')
(ROOT / "assets/layout.svg").write_text("\n".join(svg))
