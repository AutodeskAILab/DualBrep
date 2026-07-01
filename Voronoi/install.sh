#!/usr/bin/env bash
# Set up the build dependencies for the `calculate_voronoi` tool.
#
#   GTE and vcglib are header-only libraries (formerly git submodules) cloned
#   into external/. vcpkg provides the compiled dependencies
#   (CGAL / OpenCASCADE / Geogram / glog / Ceres / argparse).
#
# Run once from this directory, then build with the commands printed at the end.
set -euo pipefail
cd "$(dirname "$0")"

# System packages needed to build the dependencies.
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build pkg-config \
    libgl1-mesa-dev libglu1-mesa-dev libx11-dev gfortran libfontconfig-dev \
    curl zip unzip tar

mkdir -p external

# Header-only libraries.
[ -e external/GTE/GTE ]    || git clone --depth 1 https://github.com/davideberly/GeometricTools external/GTE
[ -e external/vcglib/vcg ] || git clone --depth 1 https://github.com/cnr-isti-vclab/vcglib     external/vcglib

# vcpkg + compiled C++ dependencies.
[ -d external/vcpkg/.git ] || git clone https://github.com/microsoft/vcpkg external/vcpkg
[ -x external/vcpkg/vcpkg ] || ./external/vcpkg/bootstrap-vcpkg.sh
./external/vcpkg/vcpkg install cgal opencascade glog ceres argparse geogram

cat <<'EOF'

Dependencies installed. Build the tool with:

  cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=external/vcpkg/scripts/buildsystems/vcpkg.cmake
  cmake --build build --config Release -j

The executable will be at build/calculate_voronoi/calculate_voronoi
EOF
