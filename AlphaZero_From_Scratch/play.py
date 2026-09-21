"""Play Connect Four against the trained AlphaZero checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from alphazero_lib import ConnectFour, MCTS, ResNet

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "model_7_ConnectFour.pt"


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")

    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(game, 9, 128, device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    args = {
        "C": 2,
        "num_searches": 100,
        "dirichlet_epsilon": 0.0,
        "dirichlet_alpha": 0.3,
    }
    mcts = MCTS(game, args, model)

    state = game.get_initial_states()
    player = 1  # human starts

    print(f"device={device} | loaded {CHECKPOINT.name}")
    print("You are player 1. Enter a column 0-6.\n")

    while True:
        print(np.array(state, dtype=int))

        if player == 1:
            valid = game.get_valid_moves(state)
            cols = [i for i in range(game.action_size) if valid[i] == 1]
            print("valid columns:", cols)
            raw = input("your column: ").strip()
            try:
                action = int(raw)
            except ValueError:
                print("enter an integer")
                continue
            if action < 0 or action >= game.action_size or valid[action] == 0:
                print("illegal move")
                continue
        else:
            print("AI thinking...")
            probs = mcts.search(game.change_perspective(state, player))
            action = int(np.argmax(probs))
            print("AI plays", action)

        state = game.get_next_state(state, action, player)
        value, done = game.get_value_and_terminated(state, action)
        if done:
            print(np.array(state, dtype=int))
            if value == 1:
                print("YOU won!" if player == 1 else "AI won!")
            else:
                print("draw")
            break
        player = game.get_opponent(player)


if __name__ == "__main__":
    main()
