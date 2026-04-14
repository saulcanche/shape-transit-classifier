# Use an official Ubuntu parent image
FROM ubuntu:22.04

# Avoid tzdata interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies: C++ compiler, Make, OpenCV, GTest, Python3
RUN apt-get update && apt-get install -y \
    g++ \
    make \
    libopencv-dev \
    libgtest-dev \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Build the project
RUN make all

# Since the application might try to open GUI windows, setting the platform to offscreen
# allows it to run headlessly (e.g., for automated tests).
ENV QT_QPA_PLATFORM=offscreen

# Command to run the application by default
CMD ["./shape_classifier"]
