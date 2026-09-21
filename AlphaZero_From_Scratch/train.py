"""Train AlphaZero on Connect Four (multi-core self-play)."""

from __future__ import annotations

from pathlib import Path

import torch

from alphazero_lib import AlphaZeroParallel, ConnectFour, ResNet, available_selfplay_workers

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)


def main():
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Same architecture used for the released checkpoint
    model = ResNet(game, 9, 128, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        "C": 2,
        "num_searches": 600,
        "num_iterations": 8,
        "num_selfPlay_iterations": 1024,  # 1024 // 256 = 4 batches per iteration
        "num_parallel_games": 256,
        "num_workers": None,  # None → os.cpu_count() - 1
        "num_epochs": 4,
        "batch_size": 2048,
        "temperature": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3,
        # save into this folder
        "checkpoint_dir": str(CHECKPOINTS),
    }
    print(args)
    print("workers:", available_selfplay_workers(args["num_workers"]))

    AlphaZeroParallel(model, optimizer, game, args).learn()


if __name__ == "__main__":
    main()
