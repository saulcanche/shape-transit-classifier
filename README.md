# Shape Transit Classifier

A C++ based computer vision application that processes a video stream to detect and classify shapes as they transit through the frame.

It utilizes image processing techniques including contour extraction, Hu Moments, and Discrete Fourier Transform (DFT) descriptors via OpenCV to accurately identify and classify various shapes.

## Prerequisites

**Tip: Using Docker is the absolute easiest way to run this project!** If you have Docker installed, you can skip installing local dependencies entirely. See the [Docker Quickstart](#docker-quickstart-recommended) section below.

**For native installation:**
- **C++17** compatible compiler (e.g., GCC or Clang)
- **OpenCV 4** (`libopencv-dev`)
- **Google Test** (`libgtest-dev`) for unit testing
- **Python 3** (for the precision calculation test script)

*(See the [Docker Quickstart](#docker-quickstart-recommended) section below for the containerized approach.)*

## Building the Project

The project includes a `Makefile` for easy compilation.

To build the main application:

```bash
make all
```

This will produce an executable named `shape_classifier`.

## Usage

Run the classifier with:

```bash
./shape_classifier
```

*(Note: Ensure required `data` resources are present for reference images or video inputs as expected by `main.cpp`.)*

## Testing

The project contains unit tests to verify individual components and an end-to-end precision test.

To build and run all tests:

```bash
make test
```

This command will:
1. Compile and run the Google Test suite (`run_tests`).
2. Run the shape classifier in headless mode (`QT_QPA_PLATFORM=offscreen`) to generate classification output.
3. Execute `scripts/calculate_precision.py` to evaluate the classification accuracy against the expected results.

## Docker Quickstart (Recommended)

Running with Docker is the smoothest way to try out this project because it handles all the heavy C++ and OpenCV dependencies for you! You don't need to install anything locally except Docker itself.

To build the Docker image:

```bash
docker build -t shape-transit-classifier .
```

To run the full test suite inside Docker:

```bash
docker run --rm shape-transit-classifier make test
```

*(Note: The Docker container defaults to headless mode (`QT_QPA_PLATFORM=offscreen`) to allow the application and tests to run without requiring X11 display forwarding.)*

## Cleanup

To remove compiled binaries:

```bash
make clean
```
