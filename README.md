# find_packages — Zero-Shot Visual Object Finder

Findet ein Referenz-Objekt in einem größeren Suchbild — **ohne Training, ohne Fine-Tuning**.

```
YOLO (yolo11n)    →  erkennt Kandidaten im Suchbild
DINOv2 (ViT-B)   →  welcher Kandidat ähnelt dem Referenzbild am meisten?
Cosine Similarity →  grüne Bounding Box um den besten Treffer
```

Laufzeit: **~5 Sekunden** auf Apple Silicon (MPS). Erster Run lädt Modelle einmalig herunter.

Entwickelt für Bücher als Testcase, Ziel-Usecase: **DHL-Pakete in Rollcontainern finden**.

---

## Setup

```bash
git clone git@github.com:Pavlos-96/find_packages.git
cd find_packages
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Benutzung

Bilder ablegen:

```
reference/target.jpg   # Foto des gesuchten Objekts (selbst gecroppt, kein Hintergrund)
scene/search.jpg       # Suchbild (z.B. Regal, Rollcontainer)
```

Starten:

```bash
python main.py
```

Ergebnis: `output/result.jpg` — grüne Box = bester Treffer, graue Boxen = weitere YOLO-Kandidaten.

### Optionen

```bash
python main.py --reference reference/target.jpg   # anderes Referenzbild
python main.py --scene scene/other.jpg            # anderes Suchbild
python main.py --yolo-class -1 --conf 0.05        # alle YOLO-Klassen, niedrigere Schwelle
python main.py --output output/mein_ergebnis.jpg  # anderer Ausgabepfad
```

| Flag | Default | Bedeutung |
|------|---------|-----------|
| `--yolo-class` | `73` (book) | COCO-Klasse für Kandidaten-Erkennung, `-1` = alle |
| `--conf` | `0.15` | YOLO-Confidence-Schwelle |

---

## Wie es funktioniert

1. **YOLO** erkennt Objekte der gewählten Klasse im Suchbild → typisch 5–25 Kandidaten
2. **DINOv2** berechnet ein Embedding für das Referenzbild und für jeden Kandidaten-Crop
3. **Cosine Similarity** wählt den ähnlichsten Kandidaten aus
4. Ergebnis wird ins Suchbild gezeichnet

---

## Nächste Schritte

- **DHL-Pakete:** YOLO kennt keine Pakete. Lösung: ~30 annotierte Fotos + kurzes Fine-Tuning (~20 Min) → eigene YOLO-Klasse `package`. DINOv2-Teil bleibt unverändert.
- **Mehrere Referenzbilder:** mehrere Fotos in `reference/` ablegen → Score = Maximum über alle Referenzen (noch nicht implementiert).
- **SAM als Alternative:** Segment Anything als Kandidaten-Detektor statt YOLO, falls kein Fine-Tuning möglich.
