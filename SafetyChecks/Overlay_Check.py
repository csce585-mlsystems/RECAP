from PIL import Image
import numpy as np

p = "/Users/aryananand/Downloads/train 2/targets/hurricane-harvey_00000070_post_disaster_target.png"
m = np.array(Image.open(p))
print("Unique values:", np.unique(m))  # expect a subset of [0,1,2,3,4]

PALETTE = {
    0: (0, 0, 0),        # background
    1: (0, 200, 0),      # no damage
    2: (255, 215, 0),    # minor
    3: (255, 140, 0),    # major
    4: (220, 20, 60),    # destroyed
}

def colorize(mask_np):
    out = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for k, rgb in PALETTE.items():
        out[mask_np == k] = rgb
    return Image.fromarray(out)

m = np.array(Image.open(p))
colorize(m).save("info/mask_color.png")

post_path = "/Users/aryananand/Downloads/train 2/images/hurricane-harvey_00000070_post_disaster.png"
post = Image.open(post_path).convert("RGBA")
mask_rgb = colorize(m).convert("RGBA")
overlay = Image.blend(post, mask_rgb, alpha=0.45)
overlay.save("info/overlay.png")
