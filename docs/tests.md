# PRIVES - Test Suite Documentation

This document describes the architecture, coverage, and commands needed to run the unit test suite for the PRIVES project. The tests are implemented with **Catch2 (v3.7.1)** and do not require hardware dependencies (Hailo-8) to run, which allows them to be validated both locally and in Continuous Integration (CI) environments.

## Test Architecture and Portability

The build system (`CMakeLists.txt`) has been designed to make hardware drivers (HailoRT, GStreamer) **optional**. The purely algorithmic parts of the code have been isolated from the inference logic, ensuring the portability of the tests. 

The suite consists of 4 executables:

| Target | Required Dependencies | WSL / Ubuntu (CI) | RPi5 |
|:-------|:----------------------|:-----------------:|:----:|
| `test_anonymizer` | OpenCV | ✅ | ✅ |
| `test_pos` | OpenCV | ✅ | ✅ |
| `test_detector` | OpenCV | ✅ | ✅ |
| `test_estimation` | OpenCV, FFTW3, Iir1 | ✅ | ✅ |

> **Note:** The face detection module demonstration `facedetection_demo` is only built if HailoRT and GStreamer are present (only on the Raspberry Pi 5).

## Test Coverage (42 Test Cases)

All tests use synthetic in-memory generated signals to ensure deterministic behavior.

### 1. Anonymizer (`test_anonymizer`) - 8 tests
Tests the spatial permutation (scrambling) of images, verifying:
- **Spatial invariance**: The generated image maintains the same mean values per channel as the original image.
- **Permutation effectiveness**: Verifies that pixels have actually been displaced.
- **Temporal determinism**: Consecutive executions with the same seed/configuration generate byte-identical results.
- **Edge cases/Validation**: Rejection of empty images, single-channel images, invalid dimensions, or negative/zero macroblocks.
- **Automatic recalibration**: Resolution change clears the internal state.
- **Flexibility**: Validation of multiple macroblock sizes (4, 8, 16, 32).

### 2. POS Extractor (`test_pos`) - 10 tests
Tests the extraction of the photoplethysmographic pulse via POS, verifying:
- **Warm-up**: Validation of the initial period needed before yielding the first result (window filling).
- **Periodicity and Sanity**: Verification that the generated signal is periodic and lacks Inf/NaN values under normal use.
- **Edge cases**: Completely black inputs (avoid zero divisions).
- **Input validation**: Invalid, empty, or black-and-white matrices (return false).
- **API consistency**: `processSample` and `processFrame` methods produce the same signal under the same conditions.
- **Illumination Robustness**: Attenuation of artificial light variations on the generated signal.

### 3. Frequencies and Spectra (`test_estimation`) - 9 tests
Tests the IIR filtering and FFT spectrum extraction algorithms:
- **Nominal Accuracy**: Correct detection of 72 BPM / 15 RPM on a signal without perturbations.
- **Physiological HR sweep**: Response and detection across the full HR spectrum (45-170 BPM).
- **Physiological FR sweep**: Response and detection of the respiratory rate spectrum (10-30 RPM).
- **Band rejection**: Perturbations with high/low out-of-band frequencies do not affect the result.
- **Safety**: Pure Direct Current (DC) input does not crash the process.
- **Cadences**: Verification of the periodic internal return post-calibration (returns a validated analysis value every ~30 samples).

### 4. Facial Detection (`test_detector`) - 15 tests
Tests the purely algorithmic part of the detection (NMS, metrics, and spatial cropping), validating:
- **IoU Logic**: Null intersections, partial overlaps, inner boxes, and zero areas.
- **Non-Maximum Suppression (NMS)**: Suppression with multiple overlapping neighbors, preserved by order, and multiple clusters.
- **Scaling (Letterbox)**: Correct padding and scaling under square images, 16:9 widescreen, portrait, horizontal, and upscaling scenarios.
- **Output Geometry**: Verification of validation dimensions of the ROI required for `Anonymizer`.

## Configuration and Test Execution

To run the tests on your local machine or WSL, follow these steps:

### 1. Install Dependencies (Linux / WSL)
You will need build tools, OpenCV, FFTW3, and the Iir1 library.

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopencv-dev libfftw3-dev libiir-dev
```

*(Catch2 is downloaded automatically via CMake/FetchContent when generating the configuration, so there is no need to install it manually).*

### 2. Build
Generate the build directory and launch the compilation process:

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

### 3. Run Tests
You can run all tests in parallel and integrated using CTest (the default CMake tool for testing suites):

```bash
ctest --output-on-failure
```

You can also run the tests individually directly on their executable. This is useful for using `Catch2`'s own filters. Some examples:

```bash
# Run the entire anonymizer group
./test_anonymizer

# Run a group of tests by TAG
./test_estimation "[sweep]"

# List all available tests within the executable
./test_detector --list-tests
```

## Continuous Integration Execution (GitHub Actions)

The repository includes an automated *workflow* for GitHub Actions located at `.github/workflows/tests.yml`. 

* **When it runs:** Automatically when a `push` is made or a `pull_request` is created on the `main` branch.
* **Environment:** `ubuntu-latest`.
* **Process:** The action installs dependencies (`libopencv-dev`, `libfftw3-dev`, `libiir-dev`), compiles all `targets`, and finally runs `ctest`. If a test fails, the CI will report a failure on the commit.
