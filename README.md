# FaceCheck: welcoming known staff at the bookstore door with a CNN

**PDAN8412 – POE Part 3 · Louis Francois Fourie (ST10152431)**

FaceCheck is a staff-recognition system for a bookstore. A camera at the door takes a few photos of whoever walks up: known staff are welcomed by name, and everyone else is sent to reception.

At its core is a **Convolutional Neural Network (CNN) built and trained from scratch in TensorFlow**. The CNN is trained once to tell 75 people apart; its second-last layer, a list of 128 numbers called the **fingerprint** (embedding), is then used to recognise people it has never seen. New staff are therefore added by **enrolment** (storing their fingerprint), with **no retraining**.

All the code is in one Jupyter notebook: [`FaceCheck.ipynb`](FaceCheck.ipynb).

---

## Results

Final test, run **once** on 15 people who were never used for training or for any decision:

| Outcome | Result |
|---|---|
| Staff welcomed by the right name | **59.1%** |
| Staff welcomed under the wrong name | 0.3% |
| Staff sent to reception | 40.6% |
| Strangers let in (target: at most 1%) | **1.2%** |

Locked door settings: **CNN v2 fingerprints · 10 enrolment photos · 3 door photos · threshold 0.76**.

How the project got there (staff welcomed with at most 1% of strangers let in):

| Step | Staff welcomed |
|---|---|
| CNN v1 (softmax name picker) | 5.1% |
| Fix 1: centre the fingerprints | 6.3% |
| Fix 2: CNN v2 (cosine name picker + CosFace margin) | 15.8% |
| Door setup: enrol 10 photos, judge 3 photos per visit | 52.0% |
| **Final test on unseen people** | **59.1%** |

---

## How it works

### 1. Training (done once)

- **Data:** 17,534 face photos of 105 people ([Pins Face Recognition](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition), Kaggle). 138 duplicate photos are removed, leaving 17,396.
- **Split by person**, so the people used to test the door are never seen during training:

  | Group | People | Photos | Used for |
  |---|---|---|---|
  | Staff | 75 | 12,434 | Training the CNN (each person's photos split 70/15/15 into train/validation/test) |
  | Validation newcomers | 15 | 2,469 | Every door decision (model choice, door settings, threshold) |
  | Test newcomers | 15 | 2,493 | The final test, run once |

- **Input pipeline (`tf.data`):** each photo is resized to 128 × 128. Training photos are augmented at random (brightness, contrast, saturation, 10% black-and-white, mirror, small rotation, zoom and shift).
- **CNN:** rescaling → 4 convolution blocks (32 → 64 → 128 → 256 filters; each block is two 3×3 convolutions with batch normalisation and ReLU, then max pooling) → global average pooling → dropout 30% → **`fingerprint` layer (128 numbers)** → name picker for the 75 staff. About 1.2 million weights.
- **CNN v2 (the final model)** replaces the softmax name picker with a **cosine name picker and a CosFace margin** (scale 30, margin 0.15), so training compares faces the same way the door does and pushes different people further apart.

### 2. At the bookstore (every day)

1. **Cut off the name picker.** The trained layers become a *fingerprint maker*: any face photo → 128 numbers, scaled to length 1.
2. **Enrol** a new staff member: 10 photos → 10 fingerprints → averaged into one **template**, saved under their name.
3. **At the door:** 3 photos of the visitor → averaged fingerprint → compared with every template using **cosine similarity**.
4. If the best match scores **0.76 or more**: *"Welcome, name!"* Otherwise: *"Please go to reception."*

The threshold 0.76 was chosen on the validation newcomers so that at most 1% of strangers get in, and it was locked in `models/door_settings.json` before the test newcomers were used.

---

## Software and hardware requirements

| Requirement | Version used | Notes |
|---|---|---|
| Operating system | Ubuntu on WSL2 (Windows 11) | Any Linux or macOS should work; native Windows is not supported by TensorFlow GPU 2.15 |
| Python | 3.10 | Inside a conda environment |
| TensorFlow | 2.15.1 | **Needs NumPy 1.x** (see below) |
| NumPy | 1.26.4 | TensorFlow 2.15 does not work with NumPy 2 |
| pandas | 2.3.3 | |
| OpenCV (`opencv-python`) | 4.11 | Image loading, measurements, Haar-cascade face finder |
| Pillow, imagehash, matplotlib, scikit-learn, tqdm, kagglehub | see `requirements.txt` | |
| Jupyter | VS Code Jupyter extension or Jupyter Notebook / Lab | |
| GPU (optional) | NVIDIA RTX 5060, 8 GB | Only needed to **retrain** the CNNs. With the saved models the notebook also runs on a CPU |
| Kaggle account | — | Free; needed to download the dataset (see step 3 below) |

Exact package versions are in [`requirements.txt`](requirements.txt).

---

## How to run it

### 1. Get the code

```bash
git clone https://github.com/EMWCCN/pdan8412-2026-poe-part-3-convolutional-neural-networks-cnn-francois05112002.git
cd pdan8412-2026-poe-part-3-convolutional-neural-networks-cnn-francois05112002
```

### 2. Create the Python environment

```bash
conda create -n facecheck python=3.10 -y
conda activate facecheck
pip install -r requirements.txt
pip install notebook ipykernel        # only if you are not using VS Code's Jupyter support
```

**GPU (optional, Linux/WSL2 with an NVIDIA card):** install the GPU build of TensorFlow instead:

```bash
pip install "tensorflow[and-cuda]==2.15.1"
```

### 3. Set up Kaggle access (for the dataset download)

The photos are **not** stored in this repository. The notebook downloads them with `kagglehub`, which uses a Kaggle API token:

1. Sign in at [kaggle.com](https://www.kaggle.com) → **Settings** → **API** → **Create New Token**. This downloads `kaggle.json`.
2. Put it in place:

   ```bash
   mkdir -p ~/.kaggle
   mv /path/to/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

The download happens once and is cached; the notebook then creates a shortcut `data/pins` pointing at the photos.

### 4. Run the notebook

Open `FaceCheck.ipynb`, select the environment's kernel, and run all cells from top to bottom.

- **`RETRAIN = False`** (default, in the *Settings and data* cell): the notebook uses the trained models saved in `models/`, so it reproduces the results in this README without retraining. A full run takes about 20–30 minutes, mostly the exploratory analysis (duplicate search, face detection and t-SNE over 17,534 photos).
- **`RETRAIN = True`**: trains both CNNs again (about 30 minutes extra on the GPU). GPU training is never exactly repeatable, so every later number will shift slightly.
- On a new GPU the **first training epoch** can take over a minute while TensorFlow compiles GPU code; later epochs take about 12 seconds. This is normal.
- `splits.json` is loaded, not re-made, so the same people always end up in the same groups.

---

## Repository contents

```
├── FaceCheck.ipynb          # the whole project (EDA → preprocessing → CNNs → door → final test → error analysis)
├── README.md
├── requirements.txt         # package versions
├── splits.json              # which person is in which group (made once, never re-shuffled)
├── data/
│   ├── meta.csv             # one row per photo, with every EDA measurement
│   ├── eda_flags.csv        # photos flagged by the EDA checks
│   ├── staff_split.csv      # every staff photo: person, label (0–74), train/val/test
│   ├── newcomers.csv        # every validation/test newcomer photo
│   └── class_names.json     # label number → name
│   (pins/ is created by the notebook and is not in the repository)
├── models/
│   ├── face_cnn.keras               # CNN v1 (softmax name picker)
│   ├── fingerprint_maker.keras      # CNN v1 without the name picker
│   ├── training_log.csv             # CNN v1, every epoch
│   ├── mean_fingerprint.npy         # the "average face" used in Fix 1
│   ├── face_cnn_v2.keras            # CNN v2 (cosine name picker + margin) – the final model
│   ├── fingerprint_maker_v2.keras   # CNN v2 without the name picker – used at the door
│   ├── training_log_v2.csv          # CNN v2, every epoch
│   └── door_settings.json           # the locked door settings (threshold 0.76, 10 + 3 photos)
├── figures/                 # every chart the notebook saves
└── selfies/                 # image used in the notebook
```

---

## Notebook sections

1. **Imports and where we use them** – libraries, seed, folders, GPU check
2. **Download & inspect the data** – Kaggle download, folder and file counts
3. **Create the META table** – one row per photo
4. **Exploratory data analysis (EDA 1–10)** – file integrity, photos per person, sizes, pixels, shortcut check, duplicates, looking at the photos, face counts, who is in the dataset, raw pixel map
5. **EDA summary** – every finding and the decision it led to
6. **Data preprocessing part 1** – remove 138 duplicates, split by person, split staff photos 70/15/15
7. **Data preprocessing part 2** – `tf.data` pipeline and augmentation
8. **Build and train the CNN** – CNN v1, sanity check, training, staff test accuracy (75.4%)
9. **Feature extraction + embedding** – cut off the name picker, fingerprints of unseen people, fingerprint map
10. **New employees & the door threshold** – enrolment and door simulation on validation newcomers (v1 welcomes only 5.1%)
11. **Fix 1 and Fix 2** – centring (6.3%), CNN v2 with a cosine name picker and margin (15.8%, staff test accuracy 89.4%)
12. **Door settings** – 10 enrolment photos and 3 door photos (52.0%), settings locked
13. **Final test** – 15 unseen people, run once (59.1% welcomed, 1.2% strangers let in)
14. **Using the CNN like the bookstore would** – `enrol_staff()` and `door()`, with a demo
15. **Error analysis** – who the door struggles with, look-alike strangers, which photo types are hard
16. **Results summary and key findings**

---

## Known limitations

- The dataset is celebrities in professional photos taken over many years, which is harder than a fixed door camera seeing the same staff every day.
- The 3-door-photo setting averaged three *different* photos; real consecutive camera frames vary less, so that result is optimistic.
- Only 8 staff were enrolled per simulated round; a larger staff list would need the threshold re-checked.
- Most stranger mistakes came from one look-alike pair, and side-profile photos score lowest, so a real door should ask visitors to look at the camera.

---
> **If GitHub can't display the notebook** ("Unable to render code block"), click the **download** button on the notebook's page and open it in VS Code or Jupyter. All outputs are saved in the file, so it can be read without running anything.
---

**Author:** Louis Francois Fourie · ST10152431 · PDAN8412 Programming in Data Analytics 2
