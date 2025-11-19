import asyncio
import subprocess
import time
from app import run_uvicorn

ngrok_cmd = ["ngrok", "http", "--domain=enough-blatantly-whale.ngrok-free.app", "8000"]

async def run():
    if run_uvicorn() == True:
        time.sleep(2)
        print("[INFO] Starting ngrok tunnel...")
        ngrok_process = subprocess.run(ngrok_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, capture_output=True, text=True)
        print(ngrok_process.stdout)

if __name__ == "__main__":
    asyncio.run(run())