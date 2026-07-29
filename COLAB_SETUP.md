# ZetaGo Dataset Generation on Google Colab

Use this guide to run self-play game generation and dataset extraction on Colab's free GPUs.

## Step-by-Step

### 1. Open Google Colab
Go to https://colab.research.google.com and create a new notebook.

### 2. Mount Google Drive (save output to your Drive)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Clone ZetaGo repo into Drive
```bash
%cd /content/drive/MyDrive
!git clone https://github.com/rahatut/ZetaGo.git
%cd ZetaGo
```

(Or if using local copy: upload a zip, unzip it, and `cd` to it.)

### 4. Install dependencies
```bash
!pip install -q -r requirements.txt
```

### 5. Extract the Katago model
```bash
%cd katago/models
!gunzip -c g170e-b10c128-s1141046784-d204142634.txt.gz > g170e-b10c128-s1141046784-d204142634.txt
%cd /content/drive/MyDrive/ZetaGo
```

### 6. Generate self-play games (SGFs)
**Option A: Using the Katago `match` command**
```bash
!mkdir -p data/raw/sgf
!katago/bin/katago match \
  -config katago/configs/selfplay7x7_match.cfg \
  -sgf-output-dir data/raw/sgf \
  -override-config "numGamesTotal=100"
```

**Option B: Run for a fixed time, then stop**
```bash
import subprocess
import time

proc = subprocess.Popen([
    'katago/bin/katago', 'match',
    '-config', 'katago/configs/selfplay7x7_match.cfg',
    '-sgf-output-dir', 'data/raw/sgf',
    '-override-config', 'numGamesTotal=1000000'  # max, will stop early
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Let it run for 10 minutes (adjust as needed)
time.sleep(600)

# Stop gracefully
proc.terminate()
try:
    proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    proc.kill()

print(f"Generated games. Checking data/raw/sgf/...")
```

### 7. Check how many SGFs were generated
```bash
!ls -la data/raw/sgf/ | head -20
!find data/raw/sgf -name "*.sgf" | wc -l
```

### 8. Build the dataset (replay SGFs → HDF5)
```bash
!python3 data/build_dataset.py
```

This will:
- Read all `.sgf` files from `data/raw/sgf/`
- Replay each through the engine
- Extract training triples
- Write to `data/processed/train.h5` and `data/processed/val.h5`

### 9. Download the HDF5 files to your local machine
```python
from google.colab import files

files.download('data/processed/train.h5')
files.download('data/processed/val.h5')
```

Or they're already in `/content/drive/MyDrive/ZetaGo/data/processed/` if you mounted Drive.

---

## Tips

- **Free Colab**: Notebook may disconnect after ~12 hours. Save progress to Drive frequently.
- **Colab Pro/Pro+**: Longer session limits, more powerful GPUs.
- **Speed**: Even CPU-only Katago can generate 100–200 games/hour. 1000 games ≈ 5–10 hours.
- **Batch size**: Adjust `numGameThreads` in the config if Colab OOMs; try 4 or 6 instead of 12.
- **Model**: The bundled model is small and runs fine on Colab's free resources.

---

## Full Colab Notebook (Copy-Paste)

Paste this into a fresh Colab notebook:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repo
!cd /content/drive/MyDrive && git clone https://github.com/rahatut/ZetaGo.git
%cd /content/drive/MyDrive/ZetaGo

# Install deps
!pip install -q -r requirements.txt

# Extract model
!cd katago/models && gunzip -c g170e-b10c128-s1141046784-d204142634.txt.gz > g170e-b10c128-s1141046784-d204142634.txt
%cd /content/drive/MyDrive/ZetaGo

# Generate games
!mkdir -p data/raw/sgf
!katago/bin/katago match \
  -config katago/configs/selfplay7x7_match.cfg \
  -sgf-output-dir data/raw/sgf \
  -override-config "numGamesTotal=100"

# Check
!ls data/raw/sgf/ | head -10

# Build dataset
!python3 data/build_dataset.py

# Download (optional)
from google.colab import files
files.download('data/processed/train.h5')
files.download('data/processed/val.h5')

print("✓ Done! Check your Drive: /MyDrive/ZetaGo/data/processed/")
```

Run each cell and you're done!
