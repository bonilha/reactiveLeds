python3 << 'EOF'
import sounddevice as sd
import numpy as np
import time

# Record 3 seconds
print("Recording 3 seconds from device 4...")
print("Make noise into line-in NOW!")
recording = sd.rec(int(3 * 44100), samplerate=44100, channels=2, device=4, dtype='float32')
sd.wait()

# Analyze
print(f"\nMin value: {np.min(recording):.6f}")
print(f"Max value: {np.max(recording):.6f}")
print(f"Mean: {np.mean(recording):.6f}")
print(f"Std dev: {np.std(recording):.6f}")
print(f"RMS: {np.sqrt(np.mean(recording**2)):.6f}")

# Check if it's just noise
if np.std(recording) < 0.001 and np.mean(np.abs(recording)) < 0.001:
    print("\n⚠ DEAD SIGNAL - No audio detected at all")
elif np.std(recording) > 0.0001 and np.max(np.abs(recording)) < 0.05:
    print("\n⚠ NOISE PATTERN - Consistent low-level noise (wrong input source)")
elif np.max(np.abs(recording)) > 0.1:
    print("\n✓ GOOD SIGNAL DETECTED")
else:
    print("\n⚠ VERY LOW SIGNAL")

# Sample the data
print(f"\nFirst 20 samples: {recording[:20, 0]}")
EOF