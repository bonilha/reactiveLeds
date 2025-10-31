# LED Sync Refactor (Raspberry Pi)

Este pacote separa os **efeitos** do arquivo principal (`sync.py`) e centraliza utilitários em `fxcore.py`.
Inclui:
- Limitador de corrente por frame (orçamento em A) e estimativa de **I**/**P**.
- Métricas históricas (1 linha/segundo) opcionais em CSV e rolling windows (60/300/900s).
- Conjunto de efeitos limpos e reativos (spectrum, blade, dots, ripple v2, fire v3, etc.).

## Estrutura
```
fxcore.py             # Contexto com HSV vetorizado, floor, segmentação, render com power-cap
effects/
  __init__.py         # Registro dos efeitos
  basics.py           # line/mirror/rainbow/vu
  dynamics.py         # peak-hold, full-pulse, waterfall, bass-ripple v2
  clean.py            # blade, center-bloom, dots, centroid-comet, outward-burst, quantized-sections
  fire.py             # fire edge/center v3
sync.py               # Principal: RX UDP, time-sync TCP, EQ, scheduler, métricas e status
```

## Requisitos
- Python no Raspberry Pi com `neopixel`, `board`, `numpy`.
- Alimentação 5V com injeção em múltiplos pontos para tiras longas.

## Uso
```
python3 sync.py --metrics-log /var/tmp/led-metrics.csv --metrics-interval 1.0 --metrics-windows 60,300,900
```
No status:
- `I`/`P` instantâneo (EMA), `CAP` quando houver corte.
- `I1m/P1m/CAP1m` = média/máx na janela de 60s.

Ajuste o orçamento em `sync.py` (arg CLI) ou diretamente em `FXContext(current_budget_a=18.0)`.
