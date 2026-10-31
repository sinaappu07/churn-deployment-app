"""Container entry point. One image, two services, chosen with the SERVICE variable:

    SERVICE=api (default)  -> FastAPI backend
    SERVICE=ui             -> Streamlit frontend

Cloud platforms such as Render give the port in the PORT variable.
"""
import os
import sys

port = os.environ.get("PORT", "8000")

if os.environ.get("SERVICE", "api").strip().lower() == "ui":
    command = [
        sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false",
    ]
else:
    command = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", port]

print("Starting:", " ".join(command), flush=True)
os.execv(sys.executable, command)
