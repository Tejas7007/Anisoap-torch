
# UK-DALE Showcase (Quick Demo for Dr. Gupta)

This bundle helps you **show the UK-DALE dataset** quickly—either **entire dataset** (HDF5) or **one house for many days with all appliances**.

## ✅ What you’ll show
1. **Dataset overview:** houses, sampling rates, channels.
2. **House-level summary:** list of appliances and meter channels.
3. **Many days** of **mains** + **all available appliances** for a house (e.g., House 1) with clean plots.
4. A few **appliance snippets** zoomed in.

---

## 0) Recommended Environment
- Python 3.10–3.12
- Create a fresh environment:
  ```bash
  uv venv .venv && source .venv/bin/activate  # or python -m venv .venv
  pip install -U pip wheel
  pip install numpy pandas matplotlib tables h5py scipy pytz tzlocal
  # NILMTK (one of the two install routes below)
  pip install nilmtk  # often works
  # If you hit issues, try:
  # pip install git+https://github.com/nilmtk/nilmtk.git
  ```

> If `tables` (PyTables) or `numexpr` gives trouble on Apple Silicon, install via conda-forge:
> ```bash
> conda create -n nilm python=3.11 -c conda-forge pytables numexpr numpy pandas matplotlib h5py scipy pytz tzlocal
> pip install nilmtk
> ```

---

## 1) Get the data
The official page is:
- https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/UK-DALE

You have two options:

### Option A — Use the prebuilt HDF5 (fastest to demo)
Download `ukdale.h5` (if your access allows) and place it here:
```
/path/to/data/ukdale/ukdale.h5
```
Then skip to **Step 3**.

### Option B — Use raw house folders and convert to HDF5
1. Download the raw data (example for House 1 & 2):
   ```bash
   mkdir -p /path/to/data/ukdale/raw
   cd /path/to/data/ukdale/raw

   # Example sources (adjust as needed if mirrors change):
   # House 1
   wget -r -np -nH --cut-dirs=7 -R "index.html*"      https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/UK-DALE/UK-DALE-2017/low_freq/house_1/

   # House 2
   wget -r -np -nH --cut-dirs=7 -R "index.html*"      https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/UK-DALE/UK-DALE-2017/low_freq/house_2/
   ```

2. Convert to `ukdale.h5` using NILMTK's converter:
   ```bash
   python - << 'PY'
from nilmtk.dataset_converters import convert_ukdale
convert_ukdale(
    input_path="/path/to/data/ukdale/raw",
    output_filename="/path/to/data/ukdale/ukdale.h5"
)
print("Converted → /path/to/data/ukdale/ukdale.h5")
PY
   ```

---

## 2) Run the demo notebook
Open:
```
ukdale_showcase.ipynb
```
Then run all cells. It will:
- Load `/path/to/data/ukdale/ukdale.h5`
- Print dataset/house metadata
- List **all appliances** in the chosen house
- Plot **many days** of mains + all available appliances (resampled for clarity)
- Provide **zoomed** windows

**Tip:** To switch houses or date ranges, edit the `HOUSE_ID`, `START`, and `END` variables near the top.

---

## 3) (Optional) CLI: show one house quickly
If you prefer a terminal-only run (fast tables + PNGs), use:
```bash
python show_ukdale.py   --h5 /path/to/data/ukdale/ukdale.h5   --house 1   --start "2013-04-30"   --end "2013-05-10"   --out out_house1
```

This will generate:
- `out_house1/house_summary.txt` (appliances, channels)
- `out_house1/mains_1min.png`
- `out_house1/appliances_1min.png`
- `out_house1/zoom_windows/` (short snippets)

---

## 4) What to say to Dr. Gupta (script)
- "Here is UK-DALE with N houses and appliance-level channels at approximately 6-second sampling (low frequency)."
- "This is House H. We have these appliances: (the notebook prints the list)."
- "I’ve pulled many days of data for all available appliances plus mains, resampled to 1 minute for readability."
- "Here are zoomed-in windows to show appliance signatures."

If you want the full dataset, repeat the same notebook with `houses = [1,2,3,4,5]` and loop (the cells include a template).

---

## 5) Troubleshooting
- If NILMTK can't find meters, check that you're loading the correct HDF5 and that conversion succeeded.
- If plots look empty, try a different date range (use the printed `available_timeframe`).

Good luck!
— Generated 2025-11-07
