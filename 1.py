import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ===================== CONFIG =====================
# ✅ IMPORTANT: ห้ามตั้ง SRC_DIR และ DST_DIR เป็นโฟลเดอร์เดียวกัน
# เดิมโค้ดลบ DST_DIR (shutil.rmtree) ทำให้ "dataset" ต้นทางหาย แล้ว copy ไม่เจอไฟล์

# อ้างอิงโฟลเดอร์ตามตำแหน่งไฟล์สคริปต์ ป้องกันปัญหา run จากคนละ working directory
BASE_DIR = Path(__file__).resolve().parent

SRC_DIR = BASE_DIR / "dataset"              # dataset เดิม
DST_DIR = BASE_DIR / "dataset_balanced"     # ✅ dataset ใหม่ที่ balance แล้ว

MODE = "oversample"   # "oversample" หรือ "undersample"
SEED = 42

# augmentation เบา ๆ สำหรับภาพที่ถูก oversample (ช่วยลด bias/overfit)
AUGMENT_FOR_DUP = True
AUG_PROB = 1.0          # 1.0 = ทำทุกภาพที่เป็นภาพซ้ำ
MAX_ROTATE = 8          # องศา
BRIGHTNESS = 0.12       # 0.0-0.3
CONTRAST = 0.12         # 0.0-0.3
JPEG_QUALITY = 95

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ===================== UTILS =====================
def list_images(folder: Path):
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def random_augment(img: np.ndarray) -> np.ndarray:
    """Augmentation เบา ๆ: flip, brightness/contrast, rotate เล็กน้อย"""
    out = img.copy()

    # flip
    if random.random() < 0.5:
        out = cv2.flip(out, 1)

    # brightness/contrast
    alpha = 1.0 + random.uniform(-CONTRAST, CONTRAST)   # contrast
    beta = 255.0 * random.uniform(-BRIGHTNESS, BRIGHTNESS)  # brightness
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # rotate small
    if MAX_ROTATE > 0:
        angle = random.uniform(-MAX_ROTATE, MAX_ROTATE)
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return out

def write_image(dst_path: Path, img: np.ndarray):
    safe_mkdir(dst_path.parent)
    ext = dst_path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        ok = cv2.imwrite(str(dst_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    else:
        ok = cv2.imwrite(str(dst_path), img)
    return ok

# ===================== MAIN =====================
def main():
    random.seed(SEED)

    if not SRC_DIR.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {SRC_DIR}")

    classes = [d.name for d in SRC_DIR.iterdir() if d.is_dir()]
    classes.sort()
    if not classes:
        raise RuntimeError("dataset ต้องมีโฟลเดอร์ class ย่อย เช่น dataset/sedan ...")

    # อ่านไฟล์ทั้งหมดต่อคลาส
    class_files = {}
    counts = {}
    for c in classes:
        files = list_images(SRC_DIR / c)
        class_files[c] = files
        counts[c] = len(files)

    print("=== BEFORE ===")
    for c in classes:
        print(f"{c:12s}: {counts[c]}")

    # target size
    if MODE == "oversample":
        target = max(counts.values())
    elif MODE == "undersample":
        target = min(counts.values())
    else:
        raise ValueError("MODE ต้องเป็น oversample หรือ undersample")

    print(f"\nMODE={MODE} target_per_class={target}\n")

    # เตรียม output (ห้ามลบ SRC_DIR)
    if DST_DIR.resolve() == SRC_DIR.resolve():
        raise RuntimeError("❌ DST_DIR ต้องไม่เท่ากับ SRC_DIR (ไม่งั้นจะลบ dataset ต้นทางทิ้ง)")

    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    safe_mkdir(DST_DIR)

    after_counts = {}

    for c in classes:
        src_list = class_files[c]
        if len(src_list) == 0:
            print(f"⚠️ class '{c}' ว่าง ข้าม")
            continue

        dst_class_dir = DST_DIR / c
        safe_mkdir(dst_class_dir)

        # เลือกไฟล์ให้ครบ target
        if MODE == "undersample":
            selected = random.sample(src_list, k=min(target, len(src_list)))
        else:
            # oversample: ใช้ของเดิมทั้งหมด แล้วสุ่มเพิ่มจนถึง target
            selected = list(src_list)
            while len(selected) < target:
                selected.append(random.choice(src_list))

        # เขียนไฟล์ลงปลายทาง
        for i, src_path in enumerate(selected):
            # ✅ กันกรณีไฟล์ถูกลบ/ย้ายระหว่างทำงาน หรือ path เกิดจากการรันคนละโฟลเดอร์
            if not src_path.exists():
                print("⚠️ missing, skip:", src_path)
                continue

            # ตั้งชื่อใหม่กันชนกัน
            stem = src_path.stem
            ext = src_path.suffix.lower()
            dst_path = dst_class_dir / f"{stem}_{i:05d}{ext}"

            # ถ้าเป็นรูปที่ถูกสุ่มซ้ำ (oversample) ให้ augment เบา ๆ
            is_dup = MODE == "oversample" and i >= len(src_list)

            if is_dup and AUGMENT_FOR_DUP and random.random() < AUG_PROB:
                img = cv2.imread(str(src_path))
                if img is None:
                    continue
                img_aug = random_augment(img)
                ok = write_image(dst_path, img_aug)
                if not ok:
                    print("❌ write failed:", dst_path)
            else:
                # copy ตรง ๆ
                safe_mkdir(dst_path.parent)
                shutil.copy2(src_path, dst_path)

        after_counts[c] = len(selected)

    print("\n=== AFTER ===")
    for c in classes:
        print(f"{c:12s}: {after_counts.get(c, 0)}")

    print(f"\n✅ Done. Balanced dataset saved to: {DST_DIR.resolve()}")

if __name__ == "__main__":
    main()