import subprocess
import time
import os
import signal

# Paths and commands
bot_path = r"C:\Users\bruno\Documents\Bensadon\FGV\Projetos III\fgv_chatbot\demo_wrap.py"
ngrok_command = ["ngrok", "http", "--domain=enough-blatantly-whale.ngrok-free.app", "8000"]

try:
    # Start ngrok tunnel
    print("Starting ngrok tunnel...")
    ngrok_process = subprocess.Popen(ngrok_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)  # Give ngrok a few seconds to initialize

    # Start the bot
    print("Starting bot...")
    bot_process = subprocess.Popen(["python", bot_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("\n✅ Both ngrok and bot are running. Press Ctrl+C to stop.\n")

    # Keep the script running until manually stopped
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping processes...")
    # Gracefully terminate both processes
    ngrok_process.terminate()
    bot_process.terminate()

    # If they don’t exit gracefully, force kill
    time.sleep(2)
    if ngrok_process.poll() is None:
        ngrok_process.kill()
    if bot_process.poll() is None:
        bot_process.kill()

    print("✅ All processes stopped cleanly.")
except Exception as e:
    print(f"❌ Error: {e}")
