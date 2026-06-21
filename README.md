# Zero-Shot Visual Object Finder — MVP

Findet ein Referenz-Objekt in einem größeren Suchbild — **ohne Training, ohne
Fine-Tuning**. Nur vortrainierte Vision-Encoder (DINOv2 / CLIP) + Cosine-Similarity.

```
reference/  →  Encoder  →  embedding
scene/      →  sliding-window crops  →  embeddings
                cosine_similarity  →  bester Crop  →  grüne Bounding Box
output/result.jpg
```

## Setup

```bash
cd visual-object-finder
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Beim ersten Lauf lädt `transformers` das Modell (~350 MB für DINOv2-base)
automatisch herunter.

## Benutzung

```
reference/          # 1..n Fotos des gesuchten Objekts (z.B. Buchrücken)
  target.jpg
scene/
  search.jpg        # das große Suchbild (z.B. Regal)
```

```bash
python main.py                                   # nutzt reference/ und scene/search.jpg
python main.py --scene scene/regal.jpg
python main.py --reference reference/buch.jpg    # einzelnes Referenzbild
python main.py --model clip                      # CLIP statt DINOv2
```

Ergebnis: `output/result.jpg` mit grüner Box + Similarity-Score.
Schwächere Kandidaten werden dünn grau eingezeichnet (Diagnose).

## Mehrere Referenzbilder

Lege einfach mehrere Bilder in `reference/` (z.B. `front.jpg`, `side.jpg`,
`back.jpg`). Der Score pro Crop ist das Maximum über alle Referenzen.

## Erfolgstest (Bücher)

1. Ein Buch einzeln fotografieren → `reference/target.jpg`
2. Dasselbe Buch ins Regal stellen
3. Regal fotografieren → `scene/search.jpg`
4. `python main.py`

Erfolg = grüne Box liegt auf dem richtigen Buch.

## Nächste Stufe (noch nicht implementiert)

Sliding-Window durch **SAM** oder **YOLO** ersetzen. Pipeline bleibt gleich:
Detector findet Kandidaten → DINOv2 entscheidet, welcher Kandidat das Objekt ist.
