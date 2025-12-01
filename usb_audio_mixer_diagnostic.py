#!/usr/bin/env python3
"""
USB Audio Input Source Diagnostic
Checks mixer settings and tests different configurations
"""

import sounddevice as sd
import numpy as np
import subprocess
import sys
import time

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error running command: {e}"

def check_alsa_mixer():
    """Check ALSA mixer settings for the USB device"""
    print("\n" + "=" * 80)
    print("ALSA MIXER SETTINGS (hw:2)")
    print("=" * 80)
    
    # Get all controls for card 2
    output = run_command("amixer -c 2")
    print(output)
    
    print("\n" + "=" * 80)
    print("ALSA MIXER CONTENTS (hw:2)")
    print("=" * 80)
    output = run_command("amixer -c 2 contents")
    print(output)

def test_with_different_formats(device_index=4):
    """Test the device with different sample formats and configurations"""
    print("\n" + "=" * 80)
    print(f"TESTING DEVICE {device_index} WITH DIFFERENT CONFIGURATIONS")
    print("=" * 80)
    
    configs = [
        {'dtype': 'float32', 'channels': 2, 'desc': 'Float32, Stereo'},
        {'dtype': 'int16', 'channels': 2, 'desc': 'Int16, Stereo'},
        {'dtype': 'int32', 'channels': 2, 'desc': 'Int32, Stereo'},
        {'dtype': 'float32', 'channels': 1, 'desc': 'Float32, Mono'},
        {'dtype': 'int16', 'channels': 1, 'desc': 'Int16, Mono'},
    ]
    
    for config in configs:
        print(f"\n--- Testing: {config['desc']} ---")
        try:
            duration = 2
            sr = 44100
            
            recording = sd.rec(
                int(duration * sr),
                samplerate=sr,
                channels=config['channels'],
                device=device_index,
                dtype=config['dtype']
            )
            
            # Show live meter
            for i in range(duration * 2):
                sd.wait(int(sr / 2))
                if i * sr // 2 < len(recording):
                    chunk = recording[max(0, i * sr // 2 - sr // 2):i * sr // 2]
                    if len(chunk) > 0:
                        if config['dtype'] == 'int16':
                            chunk_normalized = chunk.astype(np.float32) / 32768.0
                        elif config['dtype'] == 'int32':
                            chunk_normalized = chunk.astype(np.float32) / 2147483648.0
                        else:
                            chunk_normalized = chunk
                        
                        level = np.max(np.abs(chunk_normalized))
                        rms = np.sqrt(np.mean(chunk_normalized ** 2))
                        bars = int(level * 30)
                        sys.stdout.write(f"\r  {'█' * bars}{' ' * (30 - bars)} Peak: {level:.4f} RMS: {rms:.6f}")
                        sys.stdout.flush()
            
            sd.wait()
            
            # Analyze
            if config['dtype'] == 'int16':
                recording = recording.astype(np.float32) / 32768.0
            elif config['dtype'] == 'int32':
                recording = recording.astype(np.float32) / 2147483648.0
            
            peak = np.max(np.abs(recording))
            rms = np.sqrt(np.mean(recording ** 2))
            
            print(f"\n  Peak: {peak:.6f}, RMS: {rms:.6f}")
            
            # Check for noise patterns
            if rms > 0.0001 and peak < 0.02:
                print(f"  ⚠ NOISE PATTERN DETECTED - low peak but consistent RMS")
            elif peak > 0.1:
                print(f"  ✓ Good signal detected")
            else:
                print(f"  ⚠ Very low signal")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")

def check_pulseaudio_sources():
    """Check PulseAudio source configuration"""
    print("\n" + "=" * 80)
    print("PULSEAUDIO SOURCES")
    print("=" * 80)
    
    output = run_command("pactl list sources")
    print(output)

def test_raw_alsa_record():
    """Test recording directly with ALSA (bypassing sounddevice)"""
    print("\n" + "=" * 80)
    print("TESTING RAW ALSA RECORDING (hw:2,0)")
    print("=" * 80)
    print("Recording 3 seconds with arecord...")
    print("Make noise into your line-in NOW!")
    
    # Record to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name
    
    cmd = f"arecord -D hw:2,0 -f S16_LE -r 44100 -c 2 -d 3 {temp_file} 2>&1"
    output = run_command(cmd)
    print(output)
    
    # Analyze the file
    try:
        import wave
        with wave.open(temp_file, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            peak = np.max(np.abs(audio))
            rms = np.sqrt(np.mean(audio ** 2))
            
            print(f"\nALSA Recording Analysis:")
            print(f"  Peak: {peak:.6f}")
            print(f"  RMS:  {rms:.6f}")
            
            if rms > 0.0001 and peak < 0.02:
                print(f"  ⚠ NOISE PATTERN - This suggests wrong input source selected")
            elif peak > 0.1:
                print(f"  ✓ Good signal via raw ALSA")
            else:
                print(f"  ⚠ Very low signal via raw ALSA")
    except Exception as e:
        print(f"Error analyzing ALSA recording: {e}")
    finally:
        import os
        try:
            os.unlink(temp_file)
        except:
            pass

def suggest_mixer_fixes():
    """Suggest common mixer fixes"""
    print("\n" + "=" * 80)
    print("SUGGESTED FIXES")
    print("=" * 80)
    
    print("\n1. Try setting the capture source to Line In:")
    print("   amixer -c 2 sset 'Capture' cap")
    print("   amixer -c 2 sset 'Capture' 80%")
    
    print("\n2. If there's a 'Input Source' control, set it:")
    print("   amixer -c 2 sset 'Input Source' 'Line'")
    
    print("\n3. Unmute and set gain:")
    print("   amixer -c 2 sset 'Mic' unmute")
    print("   amixer -c 2 sset 'Mic' 80%")
    
    print("\n4. Try using alsamixer GUI to inspect/change settings:")
    print("   alsamixer -c 2")
    print("   (Use F4 to select capture, F6 to select card)")
    
    print("\n5. Check if PulseAudio is interfering:")
    print("   pactl list sources | grep -A 20 'ICUSB'")
    
    print("\n6. Reset the USB device:")
    print("   sudo usbreset $(lsusb | grep -i 'audio' | awk '{print $6}')")
    print("   (Or just unplug/replug the USB device)")
    
    print("\n7. Try different USB ports")
    
    print("\n8. Check dmesg for USB errors:")
    print("   dmesg | tail -50")

def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " USB AUDIO INPUT SOURCE DIAGNOSTIC ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Check ALSA mixer
    check_alsa_mixer()
    
    # Test with different formats
    test_with_different_formats(device_index=4)
    
    # Test raw ALSA
    test_raw_alsa_record()
    
    # Check PulseAudio
    check_pulseaudio_sources()
    
    # Suggestions
    suggest_mixer_fixes()
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()