# 비율 낮은 순으로 파일명 출력
import os
import numpy as np
from PIL import Image

mask_dir = ".\\data\\train\\masks"
results = []

for fname in os.listdir(mask_dir):
    mask = np.array(Image.open(os.path.join(mask_dir, fname)).convert("L"))
    ratio = (mask > 127).sum() / mask.size
    results.append((ratio, fname))

results.sort()
print("경로 비율 낮은 순 TOP 10:")
for ratio, fname in results[:10]:
    print(f"  {fname}: {ratio:.4f}")

print(f"평균 경로 비율: {np.mean(ratios):.4f}")
print(f"최소: {np.min(ratios):.4f} / 최대: {np.max(ratios):.4f}")
print(f"권장 pos_weight: {(1 - np.mean(ratios)) / np.mean(ratios):.1f}")