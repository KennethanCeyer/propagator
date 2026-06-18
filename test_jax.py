import jax
import time
import os

print(f"[{os.getpid()}] Starting test script.")
devices = jax.devices()
print(f"Devices: {devices}")
print("Sleeping for 10 seconds to hold the lock...")
time.sleep(10)
print("Done.")
