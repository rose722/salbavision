# Python Virtual Environment Setup

Set up an isolated Python environment for the detection scripts so dependencies don't conflict with other projects.

---

## Step 1 — Create the virtual environment

Run once from the `salbavision/` project root:

```bash
python -m venv venv
```

This creates a `venv/` folder in the project directory.

---

## Step 2 — Activate the virtual environment

You must activate it every time you open a new terminal.

**Windows (Command Prompt):**
```cmd
.\venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Git Bash):**
```bash
source venv/Scripts/activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

When active, your terminal prompt will show `(venv)` as a prefix.

---

## Step 3 — Install dependencies

With the venv active:

```bash
pip install -r requirements.txt
```

### GPU install (optional, NVIDIA CUDA only)

If you have an NVIDIA GPU, replace the `inference` install with `inference-gpu` and add PyTorch with CUDA:

```bash
pip install inference-gpu opencv-python pillow flask supabase pygame
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> Change `cu118` to match your CUDA version (e.g. `cu121` for CUDA 12.1).
> Check your version with: `nvidia-smi`

---

## Step 4 — Run a script

With the venv still active:

```bash
py drowning.py
python rf_stream_bridge.py
python stream_bridge.py
```

---

## Deactivate

When you're done:

```bash
deactivate
```

---

## Notes

- The `venv/` folder is local only — never commit it to git. Add it to `.gitignore` if not already there.
- If you see `ModuleNotFoundError`, your venv is likely not activated. Activate it and re-run.
- If you accidentally install packages without the venv active, just activate it and run `pip install -r requirements.txt` again inside it.
