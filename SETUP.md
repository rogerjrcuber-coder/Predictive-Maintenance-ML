# Setup

## Requirements

- Windows with Python installed and available through the `py` launcher
- PowerShell

Run the following commands from the project folder, the directory containing `requirements.txt`.

## Create and install

Create a virtual environment named `venv`, activate it, and install the project dependencies:

```powershell
py -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The execution policy change applies only to the current PowerShell window. When the environment is active, the prompt begins with `(venv)`.

For later sessions, open PowerShell in the project folder and activate the existing environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Run the API

With `venv` active:

```powershell
uvicorn app:app --reload
```

Open the interactive API documentation at <http://127.0.0.1:8000/docs>. On Windows systems where Application Control blocks Uvicorn's reload subprocess, do not add `--reload`.

## Run the dashboard

Open a second PowerShell window in the project folder, activate `venv` as above, then run:

```powershell
streamlit run dashboard.py
```

## Run tests

With `venv` active:

```powershell
python -m unittest discover -s tests -v
```
