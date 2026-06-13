# GrobsAI-Complete

Monorepo containing Backend and Frontend for the GrobsAI project.

Structure:
- `Backend/` - Python backend application and tests.
- `Frontend/` - JavaScript frontend app (Vite/React).

Getting started (local):

1. Backend: create a Python virtualenv and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r Backend/requirements.txt
```

2. Frontend: install node dependencies and run dev server:

```bash
cd Frontend
npm install
npm run dev
```

Prepare repository for GitHub:

- Ensure a remote is created on GitHub, then run:

```bash
git remote add origin <your-repo-url>
git push -u origin main
```

See `Backend/README.md` and `Frontend/README.md` for more details.
