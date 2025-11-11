# Reactive LEDs (PC + Raspberry Pi)

Pipeline de áudio reativa com PC (captura/processing) e Raspberry Pi (render de LEDs).

## Novidades
- **MEL** com normalização de área (default ON) e **tilt** configurável (`--mel-tilt`, default `-0.25`).
- **Reset remoto (B1)**: `pc-audio.py` envia no start (opcional) e via `POST /api/reset`; `sync.py` aplica em runtime ou pela tecla `r`.
- **Log em linha única** no `sync.py` (default). Mude com `--log-mode multi`.
- **Calibração de silêncio (5s)** efetiva: thresholds globais e **perfil de silêncio por banda** (opcional) com persistência em disco.

## Estrutura
```
fxcore.py    # Contexto HSV, piso dinâmico, render com power-cap
sync.py      # RPi: RX UDP, reset, efeitos, status/log single-line
pc-audio.py  # PC: captura (WASAPI loopback), LOG/MEL, AutoMode, B0, B1, calibração e noise profile
```

## Requisitos no Raspberry Pi
```bash
sudo /opt/led-env/bin/python3 -m pip install --upgrade pip
sudo /opt/led-env/bin/python3 -m pip install rpi_ws281x adafruit-circuitpython-neopixel
sudo /opt/led-env/bin/python3 -m pip install --force-reinstall Adafruit-Blinka
```

## Execução
### RPi
```bash
python3 sync.py --log-mode single   # default: single; use --log-mode multi para múltiplas linhas
```
- Teclas: `n`/`p` (efeitos), `r` (reset), `q` (sair)

### PC (exemplo MEL + tilt + noise profile)
```bash
python pc-audio.py --bind 0.0.0.0 --port 8000 \
  --scale mel --mel-bands 150 --mel-tilt -0.25 \
  --pkt a2 --tx-fps 75 \
  --signal-hold-ms 600 --vis-fps 60 \
  --norm-peak-ema 0.25 \
  --noise-profile --noise-headroom 1.12 --min-band 0
```

## Calibração de silêncio (5s)
- Dispare via REST:
```bash
curl -X POST localhost:8000/api/mode -H 'Content-Type: application/json' \
  -d '{"mode":"calibrate_silence","duration_sec":5}'
```
- Durante 5s o PC coleta `avg/rms` e, se `--noise-profile` estiver ligado, amostras de **todas as bandas**.
- Ao final, aplica:
  - `th_avg`/`th_rms` (percentil 80 do silêncio com headroom) e `resume_factor` automático;
  - **perfil de ruído por banda** (percentil 90) com **persistência** em `~/.reactiveleds/noise-profile-sr<SAMPLE>-nb<BANDS>.npy`.
- Veja o progresso/resultado em `/api/status` (campos `calibrating`, `calib_eta`, `last_calib`, `noise_profile_*`).

## Dicas
- Se ainda houver cintilação em silêncio: aumente `--noise-headroom` (1.15–1.20) ou defina `--min-band` (5–15).
- Para validar hardware, habilite `ENABLE_SMOKE_TEST = True` no `sync.py`.
