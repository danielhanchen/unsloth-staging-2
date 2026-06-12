"""Staging-only probe: print the live Apple GPU sensor readings.

Not part of PR #6187. Exists so the CI log shows the actual values
(or None) returned by apple.read_gpu_temperature_c / read_gpu_power_w on a
GitHub-hosted macOS runner, which tells us whether AppleSMC / IOReport are
reachable inside the virtualized runner at all. Never fails the job.
"""

import platform
import time

from utils.hardware import apple

print("platform:", platform.platform())
print("machine:", platform.machine())

temp = apple.read_gpu_temperature_c()
power_baseline = apple.read_gpu_power_w()  # first call only sets the baseline
time.sleep(0.4)
power = apple.read_gpu_power_w()

print("temperature_c:", temp)
print("power_w (baseline call, expected None):", power_baseline)
print("power_w (second call):", power)

sensors_available = (temp is not None) or (power is not None)
print("SENSORS_AVAILABLE:", sensors_available)
