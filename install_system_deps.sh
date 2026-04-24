#!/usr/bin/env bash
# install_system_deps.sh — install the Linux/WSL2 system libraries that
# PyQt5 (used by the vispy examples) needs at runtime.
#
# Usage:
#   ./install_system_deps.sh
#
# Safe to run multiple times — apt skips packages that are already installed.

set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
    echo "Error: apt-get not found. This script is for Debian/Ubuntu/WSL2 only." >&2
    echo "On other distros, install the equivalent xcb / xkbcommon / dbus libs"  >&2
    echo "using your package manager."                                           >&2
    exit 1
fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        echo "Error: must run as root or have sudo installed." >&2
        exit 1
    fi
fi

PACKAGES=(
    libxcb-xinerama0
    libxcb-cursor0
    libxcb-icccm4
    libxcb-image0
    libxcb-keysyms1
    libxcb-randr0
    libxcb-render-util0
    libxcb-shape0
    libxcb-sync1
    libxcb-xfixes0
    libxcb-xkb1
    libxkbcommon-x11-0
    libxkbcommon0
    libdbus-1-3
)

echo "Updating apt package index..."
$SUDO apt-get update

echo "Installing Qt/xcb runtime libraries..."
$SUDO apt-get install -y "${PACKAGES[@]}"

echo
echo "Done. You can now run the vispy examples, e.g.:"
echo "    python examples/yarn_freefall_vispy.py"
