# Reactive LEDs – PC (captura) + Raspberry Pi (render)

Sistema de visualização de áudio em tempo real com **captura no PC** (Windows, WASAPI loopback) e **render em LED** no Raspberry Pi (WS2812/NeoPixel).

---
## Visão geral
- **pc-audio.py (PC/Windows/Linux)**: captura áudio, calcula bandas (LOG ou **MEL**), faz *gating* (silêncio/retomada), detecta batida simples e envia frames por UDP para o RPi. Possui **servidor web** com WebSocket (/ws) e REST (/api/status, /api/mode, /api/reset).
- **sync.py (Raspberry Pi)**: recebe frames, aplica efeitos/cores, limita corrente, e exibe **status** (linha única por padrão). Aceita **B0 (config)**, **A1/A2 (áudio)** e **B1 (reset)**.
- **fxcore.py**: utilitários de cor, segmentação, render e estimativa de consumo.

Arquitetura de pacotes UDP:
- `A1`: `[0xA1][8 ts_pc][bands][beat][transition]`
- `A2`: `[0xA2][8 ts_pc][bands][beat][transition][dyn_floor][kick]`
- `B0`: `[0xB0][ver][nb_lo][nb_hi][fps_lo][fps_hi][hold_lo][hold_hi][vis_lo][vis_hi]`
- `B1`: `[0xB1]` (reset)

---
## Instalação

### Raspberry Pi
1. **Sistema**: Raspberry Pi OS (32/64‑bit). Ative I2C/SPI conforme o *driver* do NeoPixel (biblioteca `rpi_ws281x` via `neopixel`).
2. **Dependências**:
```bash
sudo apt update
sudo apt install -y python3-pip python3-numpy
sudo pip3 install rpi_ws281x adafruit-circuitpython-neopixel Adafruit-Blinka
```
3. **Fiação**: LED DIN no **GPIO18 (PWM)** (padrão do script), GND comum, 5V com **injeção em múltiplos pontos** para tiras longas.
4. **Teste de hardware** (opcional): edite `sync.py` e defina `ENABLE_SMOKE_TEST = True`; rode para ver R/G/B/White/Black.

### PC (Windows recomendado)
1. **Python 3.10+** instalado e no PATH.
2. **Dependências**:
```bash
pip install numpy sounddevice aiohttp
```
3. **WASAPI loopback**: o `sounddevice` usa *loopback* quando possível. Ajuste o *device* com `--device` se precisar.

---
## Execução

### Passo 1 — RPi (renderer)
```bash
python3 sync.py --log-mode single    # ou --log-mode multi
```
**Teclas**: `n` (próximo efeito), `p` (anterior), `r` (reset), `q` (sair).

### Passo 2 — PC (captura + processamento)
Exemplo usando **MEL** (150 bandas), tilt de graves e perfil de ruído:
```bash
python pc-audio.py --bind 0.0.0.0 --port 8000   --scale mel --mel-bands 150 --mel-tilt -0.25   --pkt a2 --tx-fps 75   --signal-hold-ms 600 --vis-fps 60   --norm-peak-ema 0.25   --noise-profile --noise-headroom 1.12 --min-band 0
```
Acesse o monitor em `http://<PC>:8000/`. JSON: `http://<PC>:8000/api/status`.

---
## Calibração de silêncio (5s)
1. **Ambiente silencioso** (sem música e, idealmente, sem ventiladores próximos do microfone).
2. Dispare a calibração:
```bash
curl -X POST http://<PC>:8000/api/mode -H 'Content-Type: application/json'   -d '{"mode":"calibrate_silence","duration_sec":5}'
```
3. O `pc-audio.py` coleta `avg/rms` (filtrados) e, se `--noise-profile` estiver ligado, **amostras por banda**.
4. Ao concluir, aplica:
   - `th_avg`/`th_rms` (percentil 80 do silêncio + *headroom*),
   - `resume_factor` (
     ~ contraste música/silêncio com *clamp* 1.4–4.0),
   - **perfil de ruído por banda** = percentil 90 (opcional), salvo em `~/.reactiveleds/noise-profile-sr<SAMPLE>-nb<BANDS>.npy`.
5. O front/WS expõe `calibrating`, `calib_eta`, `last_calib` e `noise_profile_active`.

**Se os LEDs piscarem em silêncio**:
- Aumente `--noise-headroom` para 1.15–1.20 (subtrai mais do piso),
- Ajuste `--min-band` (5–15) para um brilho mínimo quando ativo.

---
## Reset remoto (B1)
- O PC pode enviar **B1** no **start** (desligável com `--no-reset-on-start`) e a qualquer momento via:
```bash
curl -X POST http://<PC>:8000/api/reset
```
- O `sync.py` registra “`[RST] B1 recebido`” e aplica: apagar LEDs, limpar buffers e reaplicar paleta/efeito.

Se o reset não ocorrer:
- Confirme IP/porta no PC (`--raspberry-ip`, `--udp-port`).
- Verifique firewall no PC/RPi (UDP 5005).
- Veja no RPi se aparece `B1 recebido`. Caso contrário, teste envio manual:
```bash
echo -ne "±" | nc -u -w1 <RPi-IP> 5005
```

---
## Parâmetros principais
### pc-audio.py
- `--scale log|mel` / `--mel-bands N` / `--mel-tilt T` / `--mel-no-area-norm`
- `--pkt a1|a2` / `--tx-fps FPS` / `--signal-hold-ms MS` / `--vis-fps FPS`
- *Gating*: `--silence-bands`, `--silence-rms`, `--silence-duration`, `--resume-factor`, `--resume-stable`
- *Norm*: `--norm-peak-ema`
- *Noise profile*: `--noise-profile`, `--noise-headroom`, `--min-band`, `--noise-profile-path`
- *Reset*: `--no-reset-on-start`

### sync.py
- `--log-mode single|multi`

---
## Troubleshooting
- **LEDs iniciais apagados** com MEL: mantenha normalização de área e ajuste `--mel-tilt` (-0.2 a -0.35).
- **Cintilação em silêncio**: calibre (5s), aumente `--noise-headroom`, use `--min-band` > 0.
- **Sem áudio**: revise *device* (`--device`), habilite WASAPI loopback (Windows), ou use o *default input* como fallback.
- **Alimentação**: injete 5V em múltiplos pontos; ative o *current cap* no `FXContext` (já habilitado) se necessário.

---
## Segurança elétrica
Tiras longas consomem **corrente alta**. Dimensione fonte, bitolas, e aterre o GND com o RPi. O `FXContext` limita a corrente efetiva por frame e informa `I/P` aproximados.
