"""Lance l'app Streamlit et l'expose publiquement via ngrok (utile dans Google Colab).

Usage :
    export NGROK_AUTHTOKEN=...
    python run_ngrok.py
"""
import os
import subprocess
import sys
import time

from pyngrok import conf, ngrok

from src.config import NGROK_TOKEN_ENV, ROOT_DIR, STREAMLIT_PORT


def run_app() -> None:
    token = os.environ.get(NGROK_TOKEN_ENV)
    if not token:
        raise RuntimeError(f"Définissez la variable d'environnement {NGROK_TOKEN_ENV}")
    ngrok.set_auth_token(token)
    ngrok.kill()  # ferme les tunnels existants
    conf.get_default().region = "us"

    streamlit_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", str(STREAMLIT_PORT), "--server.headless", "true"],
        cwd=ROOT_DIR,
    )
    time.sleep(5)  # laisse à Streamlit le temps de démarrer

    public_url = None
    try:
        public_url = ngrok.connect(STREAMLIT_PORT)
        print(f"Streamlit app running at: {public_url}")
        streamlit_proc.wait()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"An error occurred: {exc}")
    finally:
        if public_url:
            ngrok.disconnect(public_url)
        ngrok.kill()
        streamlit_proc.terminate()


if __name__ == "__main__":
    run_app()
