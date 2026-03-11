"""main.py — Entry point for the yarn simulation.

Usage
-----
    python main.py              # auto-select GPU/CPU
    python main.py --cpu        # force CPU
    python main.py --fps 30     # lower frame rate
"""

import argparse
import warp as wp

from simulation import Simulation
from visualization import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Yarn simulation (NVIDIA Warp PBD)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU device")
    parser.add_argument("--fps", type=int, default=60, help="Target frame rate")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = "cpu" if args.cpu else None   # None = auto (GPU if available)

    wp.init()
    print(f"Warp {wp.__version__} | CUDA available: {wp.is_cuda_available()}")

    sim = Simulation(device=device)
    print(f"Running on: {sim.device} | particles: {sim.pos.shape[0]}")

    viz = Visualizer(sim, fps=args.fps)
    viz.run()


if __name__ == "__main__":
    main()
