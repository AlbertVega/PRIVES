# PRIVES
Private rPPG-based Inspection of Vital Signs for Embedded Systems

## Getting Started (Debian / Raspberry Pi OS / Ubuntu)

### 1. Previous Requirements 

Install the build tools, CMake and the development library of OpenCV:

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
```

From the root of the project, generate the configuration with CMake and compile:

```bash
# Create and access the build directory
mkdir build && cd build

# Configure the project
cmake ..

# Compile
cmake --build .
```

## Testing

For detailed information about the test suite (Catch2), test coverage, how to run tests locally, and the GitHub Actions CI setup, please refer to the [Test Suite Documentation](docs/tests.md).
