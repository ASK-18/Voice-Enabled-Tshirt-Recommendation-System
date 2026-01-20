import os
import csv
import random

IMAGE_DIR = "data/images"
OUTPUT_CSV = "data/products.csv"

colors = ["black", "white", "blue", "red", "grey"]
styles = ["casual", "minimal", "graphic"]
fits = ["regular", "oversized"]

rows = []
idx = 1

for img in os.listdir(IMAGE_DIR):
    if img.lower().endswith((".jpg", ".png", ".jpeg")):
        color = random.choice(colors)
        style = random.choice(styles)
        fit = random.choice(fits)
        price = random.choice([599, 699, 799, 899])

        rows.append([
            idx,
            f"{color.capitalize()} {style.capitalize()} T-Shirt",
            color,
            fit,
            style,
            price,
            img
        ])
        idx += 1

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["id", "title", "color", "fit", "style", "price", "image"]
    )
    writer.writerows(rows)

print("✅ products.csv generated successfully")
