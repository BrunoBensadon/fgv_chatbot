import asyncio
import subprocess
import httpx

async def wait_until_online(url: str, timeout: int = 30):
    """Poll the given URL until a response is received or timeout occurs."""
    async with httpx.AsyncClient() as client:
        for _ in range(timeout * 2):  # check every 0.5 seconds
            try:
                r = await client.get(url)
                if r.status_code < 500:
                    print(f"[ OK ] Server online at {url}")
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.5)

    print("[ ERROR ] Timeout while waiting for server to go online.")
    return False


async def run_bot():
    # 1. Start Uvicorn in a background process
    print("[INFO] Starting uvicorn...")
    uvicorn_process = subprocess.Popen(
        ["python", "-c", "from app import run_uvicorn; run_uvicorn()"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # 2. Wait for the app to be online
    print("[INFO] Waiting for server to become available...")
    await wait_until_online("http://127.0.0.1:8000")

    # 3. Run ngrok with fixed domain
    print("[INFO] Starting ngrok tunnel...")
    ngrok_cmd = [
        "ngrok", "http",
        "--domain=enough-blatantly-whale.ngrok-free.app",
        "8000"
    ]

    # Stream ngrok logs in foreground
    ngrok_process = subprocess.Popen(ngrok_cmd)

    # 4. Keep script alive until either process stops
    try:
        await asyncio.get_running_loop().run_in_executor(None, uvicorn_process.wait)
    finally:
        ngrok_process.terminate()


if __name__ == "__main__":
    asyncio.run(run_bot())
