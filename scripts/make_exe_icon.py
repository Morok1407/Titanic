"""Render the project's ship motif as a Windows icon (requires Pillow)."""

from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
out = root / "build_assets" / "titanic.ico"
out.parent.mkdir(exist_ok=True)
scale = 4
image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)
gold, navy, teal = "#e7b96a", "#0b202b", "#6b9b9c"


def points(coords):
    return [(int(x * scale), int(y * scale)) for x, y in coords]


draw.rounded_rectangle((0, 0, 255, 255), radius=56, fill=navy)
draw.polygon(points([(8, 37), (56, 37), (46, 49), (19, 49)]), fill=gold)
for rectangle in [(16, 27, 48, 35), (22, 16, 28, 25), (36, 16, 42, 25)]:
    draw.rectangle(tuple(int(value * scale) for value in rectangle), fill=gold)
draw.line(points([(12, 54), (17, 52), (22, 54), (27, 52), (32, 54), (37, 52), (42, 54), (47, 52), (52, 54)]),
          fill=teal, width=8, joint="curve")
image.save(out, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print(out)
