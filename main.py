"""Zero-Shot Visual Object Finder
Pipeline: YOLO (detect candidates) → DINOv2 (which one matches the reference)

Usage:
    python main.py                                        # defaults below
    python main.py --reference reference/target.jpg --scene scene/search.jpg
    python main.py --yolo-class 0                         # 0=person, 73=book, -1=all
    python main.py --conf 0.15                            # YOLO confidence threshold
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

# ── Models ──────────────────────────────────────────────────────────────────
DINO_MODEL   = "facebook/dinov2-base"
YOLO_MODEL   = "yolo11n.pt"   # downloaded automatically on first run (~6 MB)
YOLO_BOOK_CLASS = 73          # COCO class 73 = "book"

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_REFERENCE = "reference/target.jpg"
DEFAULT_SCENE     = "scene/search.jpg"
DEFAULT_OUTPUT    = "output/result.jpg"


# ── Device ───────────────────────────────────────────────────────────────────
def device() -> str:
    if torch.cuda.is_available():  return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


# ── Step 1: DINOv2 encoder ───────────────────────────────────────────────────
class Encoder:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModel
        self.proc  = AutoImageProcessor.from_pretrained(DINO_MODEL)
        self.model = AutoModel.from_pretrained(DINO_MODEL).to(device()).eval()

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """List of PIL images → (N, D) L2-normalised embeddings."""
        inputs = self.proc(images=[i.convert("RGB") for i in images],
                           return_tensors="pt").to(device())
        feats = self.model(**inputs).last_hidden_state[:, 0]  # CLS token
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats.cpu().numpy()


# ── Step 2: YOLO candidate detection ─────────────────────────────────────────
def detect_candidates(
    scene_path: str,
    yolo_class: int,
    conf: float,
) -> list[tuple[int, int, int, int]]:
    """Run YOLO on the scene image, return list of (x, y, w, h) boxes."""
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL)

    classes = None if yolo_class == -1 else [yolo_class]
    results = model(scene_path, conf=conf, classes=classes, verbose=False)[0]

    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes


# ── Step 3: match reference against candidates ────────────────────────────────
def find_best_match(
    ref_embedding: np.ndarray,          # (D,)
    scene: Image.Image,
    candidates: list[tuple[int, int, int, int]],
    encoder: Encoder,
) -> tuple[tuple[int, int, int, int], float, list[tuple]]:
    """
    Crop each candidate from the scene, embed with DINOv2, pick highest
    cosine similarity. Returns (best_box, best_score, all_ranked).
    """
    crops = [scene.crop((x, y, x + w, y + h)) for x, y, w, h in candidates]
    embs  = encoder.embed(crops)                       # (N, D)
    sims  = embs @ ref_embedding                       # (N,) cosine sim

    ranked = sorted(zip(sims.tolist(), candidates), reverse=True)
    best_score, best_box = ranked[0]
    return best_box, best_score, ranked


# ── Step 4: draw result ───────────────────────────────────────────────────────
def draw_result(
    scene: Image.Image,
    best_box: tuple[int, int, int, int],
    best_score: float,
    all_ranked: list[tuple],
    output_path: str,
) -> None:
    img = cv2.cvtColor(np.array(scene.convert("RGB")), cv2.COLOR_RGB2BGR)

    # All candidates in grey
    for score, (x, y, w, h) in all_ranked:
        if (x, y, w, h) == best_box:
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (140, 140, 140), 2)

    # Best match in green
    x, y, w, h = best_box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 220, 0), 4)
    label = f"{best_score:.3f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    ly = max(th + 8, y)
    cv2.rectangle(img, (x, ly - th - 8), (x + tw + 8, ly), (0, 220, 0), -1)
    cv2.putText(img, label, (x + 4, ly - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference",  default=DEFAULT_REFERENCE)
    p.add_argument("--scene",      default=DEFAULT_SCENE)
    p.add_argument("--output",     default=DEFAULT_OUTPUT)
    p.add_argument("--yolo-class", type=int, default=YOLO_BOOK_CLASS,
                   help="COCO class id (73=book, -1=all classes)")
    p.add_argument("--conf",       type=float, default=0.15,
                   help="YOLO confidence threshold (lower = more candidates)")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # Load reference
    if not os.path.exists(args.reference):
        raise FileNotFoundError(f"Referenzbild nicht gefunden: {args.reference}")
    ref_img = Image.open(args.reference).convert("RGB")
    print(f"[ref]   {args.reference}  {ref_img.size}")

    # Load scene
    if not os.path.exists(args.scene):
        raise FileNotFoundError(f"Suchbild nicht gefunden: {args.scene}")
    scene = Image.open(args.scene).convert("RGB")
    print(f"[scene] {args.scene}  {scene.size}")

    # YOLO detection
    print(f"[yolo]  detecting  (class={args.yolo_class}, conf≥{args.conf}) …")
    t1 = time.time()
    candidates = detect_candidates(args.scene, args.yolo_class, args.conf)
    print(f"        {len(candidates)} Kandidaten  ({time.time()-t1:.1f}s)")

    if not candidates:
        print("⚠  Keine Kandidaten gefunden. Versuche --yolo-class -1 oder --conf 0.05")
        return

    # DINOv2 embeddings
    print(f"[dino]  embedding reference + {len(candidates)} crops …")
    t2 = time.time()
    enc = Encoder()
    ref_emb = enc.embed([ref_img])[0]
    best_box, best_score, all_ranked = find_best_match(ref_emb, scene, candidates, enc)
    print(f"        fertig  ({time.time()-t2:.1f}s)")

    # Result
    print(f"\n[result] Score={best_score:.4f}  box={best_box}")
    for i, (s, b) in enumerate(all_ranked, 1):
        marker = "◀ BEST" if b == best_box else ""
        print(f"         #{i}  score={s:.4f}  box={b}  {marker}")

    draw_result(scene, best_box, best_score, all_ranked, args.output)
    print(f"\n[saved] {args.output}  (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
