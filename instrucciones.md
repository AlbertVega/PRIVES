# PROMPT DE EJECUCIÓN: Suite de Benchmarking de Hardware y Pruebas de Estrés para PRIVES

## 1. Contexto y Objetivo
Actúas como ingeniero de sistemas embebidos de alto rendimiento. El objetivo es diseñar, instrumentar e implementar la suite automatizada de pruebas de estrés y benchmarking de hardware sobre una plataforma Raspberry Pi 5 (8 GB RAM, ARM Cortex-A76 @ 2.4 GHz) con un acelerador Raspberry Pi AI HAT+ (Hailo-8 NPU), corriendo Raspberry Pi OS Lite (Debian 12 Bookworm, 64-bit ARM64) en modo headless.

El pipeline a evaluar está desarrollado en C++17/C++20 y comprende:
1. Captura de video en tiempo real mediante GStreamer (`v4l2src` / DMA-BUF a 30 FPS).
2. Detección facial mediante SCRFD (`scrfd_500m.hef`) en NPU Hailo.
3. Mecanismo de anonimización local $O(n)$ por permutación determinista de píxeles (Wang et al.).
4. Extracción de señal BVP con proyección ortogonal (POS) y detrending lineal.
5. Filtrado pasabanda IIR Butterworth de 4to orden (librería `Iir1`).
6. Estimación espectral de HR y RR mediante FFTW3 con ventanas deslizantes temporales.

Debes validar cuantitativamente el cumplimiento de los siguientes Requerimientos No Funcionales (RNF):
- **RNF-01 (Tasa de cuadros):** aproximadamente $\ge 30.0$ FPS sostenidos (cero caída de cuadros acumulada).
- **RNF-02 (Latencia por cuadro):** $\leq 25.0$ ms por fotograma de extremo a extremo.
- **RNF-03 (Carga de CPU):** $< 35\%$ de uso global de CPU en los 4 núcleos (sin saturación de hilos al 100%).
- **RNF-04 (Huella de RAM):** $\leq 350.0$ MB totales de memoria residente (RSS) y cero derivas (*memory leaks*).
- **RNF-13 (Potencia y Térmico):** Potencia consumida $< 12.0$ W continuos, temperatura del SoC $< 70.0$ °C y ausencia de estrangulamiento térmico (*thermal throttling*).

---

## 2. Requerimientos de Implementación

Crea dentro del repositorio una carpeta `benchmarks/` con módulos desacoplados para instrumentación, telemetría, orquestación y análisis de datos:

```text
benchmarks/
├── include/
│   └── prives_profiler.hpp      # Header C++ para instrumentación interna del pipeline
├── scripts/
│   ├── hw_monitor.py            # Daemon de monitoreo de hardware a 500 ms (CPU, RAM, Temp, Power, NPU)
│   ├── run_stress_suite.sh      # Orquestador Bash de los escenarios de prueba
│   └── check_leaks.sh           # Validación con Valgrind Massif / Heaptrack
├── analysis/
│   └── generate_report.py       # Script en Python para procesamiento de CSV, gráficos y tablas LaTeX
└── README.md                    # Instrucciones de compilación y ejecución
```

## Módulo A: Instrumentación Interna de Latencia en C++ (prives_profiler.hpp)
Implementa una clase liviana PrivesProfiler basada en `std::chrono::steady_clock` con sobrecarga casi nula (cero bloqueos en el hilo crítico de video). La clase debe medir en microsegundos ($\mu s$) cada etapa del pipeline:

   - $T_{capture}$: Tiempo de entrega de frame por GStreamer.
   - $T_{npu\_mesh}$: Inferencia tensorial en HailoRT.
   - $T_{anonymize}$: Transformación espacial / permutación de Wang et al.
   - $T_{pos\_detrend}$: Proyección de color y eliminación de tendencia lineal.
   - $T_{iir\_filter}$: Paso por los filtros pasabanda de HR y RR.
   - $T_{fftw3}$: Ejecución de la transformada de Fourier y detección de picos.
   - $T_{total\_frame}$: Latencia acumulada de extremo a extremo.

Para evitar latencia de I/O de disco durante la captura en vivo, acumula las métricas en un búfer circular preasignado en memoria RAM o un hilo desacoplado de escritura asíncrona, y vuélcalas al finalizar la prueba en benchmarks/results/pipeline_telemetry.csv con el formato:

`frame_id,timestamp_epoch_ms,t_capture_ms,t_npu_ms,t_anon_ms,t_pos_ms,t_filter_ms,t_fft_ms,t_total_ms,dropped_flag`

## Módulo B: Daemon de Monitoreo de Hardware (hw_monitor.py)

Desarrolla un script en Python (empleando `psutil`, `subprocess` y llamadas al sistema) que se ejecute en segundo plano cada 500 ms. Debe registrar en `benchmarks/results/hardware_metrics.csv`:

- **CPU y Planificación**: 
  - Porcentaje total de CPU del sistema y desglose individual por núcleo (`cpu0`, `cpu1`, `cpu2`, `cpu3`).
  - Porcentaje de CPU consumido específicamente por el PID del binario de PRIVES.
  
- **Memoria**:
  - Memoria física residente (RSS) y virtual (VMS) del proceso PRIVES en megabytes (MB) consultando `/proc/[PID]/status` (`VmRSS`, `VmSize`).
  - Uso de memoria RAM global del sistema operativo.
  
- **Térmica y Throttling**:
  - Temperatura del SoC ejecutando `vcgencmd measure_temp` (extrayendo el valor en °C).
  - Detección de throttling mediante `vcgencmd get_throttled` (decodificando el bitmask hexadecimal: `0x1` undervoltage actual, `0x2` freq capped, `0x4` throttled actual, `0x40000` throttled histórico, etc.).
  
- **Potencia Eléctrica (Vía Consultas de Software en RPi 5)**: extrae los valores eléctricos directamente del PMIC DA9091 de la Raspberry Pi 5: 
  - Ejecuta `vcgencmd pmic_read_adc` y parsea los canales `EXT5V_V` (voltaje de entrada) y `EXT5V_A` (corriente de entrada).
  - Calcula la potencia en vatios: $P = V \times I$.
 
- **Acelerador NPU Hailo**:
  - Monitorear temperatura del chip Hailo y uso de la NPU llamando a `hailortcli monitor` o mediante las APIs de estadísticas de hailort.

Formato de salida CSV: `timestamp,epoch_ms,cpu_total_pct,cpu_prives_pct,cpu_core0,cpu_core1,cpu_core2,cpu_core3,ram_rss_mb,ram_vms_mb,temp_soc_c,power_w,voltage_v,current_a,throttled_hex,npu_temp_c`

## Módulo C: Orquestador de Pruebas de Estrés (run_stress_suite.sh)
Diseña un script Bash que automatice la ejecución secuencial de los 4 escenarios de validación experimental:

- **Escenario E0: Idle Baseline (10 minutos)**:
  - Medir consumo base del sistema operativo sin el pipeline de video activo (cámara en reposo).

- **Escenario E1: Soak Test Nominal Continuo (60 minutos):**

  - Ejecución ininterrumpida de PRIVES recibiendo flujo de video a 30 FPS.
  - Evaluar la pendiente de memoria RSS (demostrar convergencia asintótica y ausencia de memory leaks) y el equilibrio térmico estacionario.

- **Escenario E2: Estrés de Escena Dinámica (15 minutos):**
  - Simulación de tracking intensivo: alternancia periódica de presencia y ausencia de rostro en la ROI para medir el coste de re-inicialización del Face Mesh y reajuste de buffers de señal.

- **Escenario E3: Sobrecarga y Contención de Memoria (10 minutos):**
  - Ejecutar PRIVES en paralelo con contención sintética del bus LPDDR4X: `stress-ng --vm 1 --vm-bytes 1G --vm-keep --timeout 10m`
  - Registrar si la saturación del bus de memoria degrada los FPS por debajo de 30 o aumenta el jitter intercuadro.
  - Incluye en `check_leaks.sh` una ejecución previa de 5 minutos instrumentada con `heaptrack` o `valgrind --tool=massif` para generar el mapa de asignaciones del heap del binario C++.

## Módulo D: Generación de Reportes y Gráficos (generate_report.py)

Desarrolla un script en Python con `pandas`, `matplotlib` y `seaborn` que procese los CSVs generados y produzca:

- **Tabla de Cumplimiento de RNFs:** Generar un archivo `rnf_summary.txt` y código de tabla en formato LaTeX (`table_rnf.tex`) que compare:
  - Métrica (FPS, Latencia ms, CPU %, RAM MB, Potencia W, Temperatura °C).
  - Umbral Teórico Requerido (RNF-01 a RNF-04, RNF-13).
  - Resultados Experimentales: Media, Desviación Estándar, Percentil 95 (P95), y Peor Caso (Worst Case).
  - Estado de Validación: CUMPLE / FALLA.
  
- Gráficos de Publicación (formato vectorial PDF):
  - `latency_breakdown.pdf`: Gráfico de áreas apiladas (Stacked Area Plot) que muestre la descomposición temporal por cuadro ($T_{npu}$, $T_{anon}$, $T_{pos}$, $T_{fft}$), con líneas horizontales de referencia en 25.0 ms (límite RNF-02) y 33.33 ms (límite de 30 FPS).
  - `memory_stability.pdf`: Serie temporal del RSS a lo largo de los 60 min del Soak Test, calculando y mostrando la pendiente de regresión lineal para certificar estabilidad de memoria.
  - `thermal_power_profile.pdf`: Gráfico con doble eje Y (Temperatura °C a la izquierda, Potencia W a la derecha) a lo largo del tiempo, contrastando contra el umbral de 70 °C y 12.0 W.
  - `interframe_jitter_hist.pdf`: Histograma con curva KDE del delta de tiempo entre fotogramas consecutivos ($\Delta t = t_i - t_{i-1}$) para verificar el determinismo de la frecuencia de muestreo de 30 Hz ($33.33 \pm \epsilon$ ms).
  
## Criterios de Aceptación del Código
- El código no debe requerir intervención manual durante la ejecución de la suite.
- Todas las rutas de salida de datos deben ser relativas al directorio `benchmarks/results/`.
- El daemon en Python debe ser robusto y capturar señales SIGINT/SIGTERM para guardar y cerrar de forma segura los archivos CSV.
- El parser de `vcgencmd pmic_read_adc` debe manejar posibles variaciones de formato o campos no presentes sin interrumpir la ejecución.
- Todo el código debe estar en inglés (comentarios solo los necesarios) y seguir las buenas prácticas de programación.