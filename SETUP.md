# Setup Guide — Study Assistant

This is the consolidated setup for a fresh local folder. If you already
completed Supabase project creation, got a Gemini API key, and installed
Tesseract OCR in an earlier setup, **you don't need to redo those** — just
reuse the same values in Step 4 below.

---

## Step 1: Put the project in one permanent folder

Pick one folder you'll keep long-term, e.g. `D:\study-assistant\`, and
extract this zip there. Going forward, don't extract future updates into a
*new* folder — replace individual files inside this same folder instead, so
you never have to rebuild your virtual environment or re-enter your `.env`
values again.

## Step 2: Install Python and create a virtual environment

1. Check Python is installed (3.10+): `python3 --version`
   If not, get it from https://www.python.org/downloads/

2. Open a terminal **inside this project folder** and run:
   ```
   python3 -m venv venv
   ```

3. **Windows only, one-time per machine** — allow PowerShell to run scripts:
   ```
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```
   (Type `Y` when prompted.)

4. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

   You'll see `(venv)` at the start of your terminal prompt when it's active.
   You need to do this every time you open a **new terminal window** — but
   not every time you edit a file.

## Step 3: Install dependencies

```
pip install -r requirements.txt
```

## Step 4: Set up your `.env` file

Copy `.env.example` to `.env` and fill in:

- `DATABASE_URL` — your Supabase connection string (Session pooler, from the
  **Connect** button in your Supabase project). Reuse your existing project
  if you have one — no need to create a new one.
- `GEMINI_API_KEY` — from https://aistudio.google.com/apikey. Reuse your
  existing key if you have one.
- `TESSERACT_CMD` (Windows only) — path to your installed Tesseract, e.g.
  `C:\Program Files\Tesseract-OCR\tesseract.exe`. Only needed if you haven't
  installed Tesseract yet — see below.

**If you don't have Tesseract OCR installed yet:** download it from
https://github.com/UB-Mannheim/tesseract/wiki (Windows), or
`brew install tesseract` (Mac) / `sudo apt install tesseract-ocr` (Linux).
Needed for scanned/image-based PDFs — regular text PDFs work without it.

## Step 5: Make sure your database is up to date

If this is a **brand-new Supabase project**: run `schema.sql` in the
Supabase SQL Editor.

If you already ran earlier migrations on an existing project, run any
`migration_vN.sql` files you haven't run yet, in order, the same way.

## Step 6: Run it

```
uvicorn main:app --reload
```

Open **http://localhost:8000**

If you don't see the gradient styling, hard-refresh your browser
(**Ctrl+Shift+R** on Windows) — browsers cache CSS aggressively, and this
usually clears it.

---

## Your day-to-day workflow from now on

```
cd D:\study-assistant
venv\Scripts\activate
uvicorn main:app --reload
```

When you get updated files from me, drop them directly into this folder
(overwriting the old ones) — don't extract into a new folder. Only re-run
`pip install -r requirements.txt` if I tell you a new dependency was added.

## Known limitations (still to build)

- No login/accounts — single-user, local only.
- No structured note generation yet.
- No coverage dashboard yet.
- No true discussion forking yet (multiple separate discussions work; branching from a point in history doesn't yet).
- Files stored on local disk, not cloud storage — fine for local dev, would need to move to Supabase Storage before deploying anywhere real.
