"""
AlphaZero (parallel MCTS) — shared library.

Kept as a real .py module so multiprocessing `spawn` workers can import it
from Jupyter/Cursor notebooks (notebook cells are not reliably picklable).
"""

from __future__ import annotations

import gc
import math
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import trange


# ---------------------------------------------------------------------------
# CPU workers — dynamic, not hard-coded to any machine
# ---------------------------------------------------------------------------

def available_selfplay_workers(requested=None):
    """
    How many self-play processes to use.

    - requested=None  → use (cpu_count - 1), minimum 1
      (leave one core for the OS / notebook / GPU driver)
    - requested=N     → clamp to [1, cpu_count]
    """
    cpu = os.cpu_count() or 1
    if requested is None:
        return max(1, cpu - 1) if cpu > 1 else 1
    return max(1, min(int(requested), cpu))


def _partition_counts(total, n_parts):
    """Split `total` games across workers as evenly as possible (no empty workers)."""
    n_parts = max(1, min(n_parts, total))
    base, rem = divmod(total, n_parts)
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def make_game(game_name: str):
    """Rebuild a game object by name inside a worker process."""
    if game_name == "ConnectFour":
        return ConnectFour()
    if game_name == "TicTacToe":
        return TicTacToe()
    raise ValueError(f"Unknown game: {game_name}")


def selfplay_worker(payload: dict):
    """
    Top-level spawn entrypoint (must live in this module for pickling).

    Rebuilds game + ResNet in the child, then calls the SAME
    AlphaZeroParallel.selfPlay() used by single-process training.
    No duplicated self-play / MCTS logic.
    """
    # Per-worker RNG so parallel games explore different lines
    seed = int(payload["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(payload["device"])
    if device.type == "cuda" and torch.cuda.is_available():
        # All workers share GPU 0; model is tiny so this is fine on a 3090
        torch.cuda.set_device(0)

    game = make_game(payload["game_name"])
    model = ResNet(
        game,
        payload["num_resBlocks"],
        payload["num_hidden"],
        device,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    # Each worker only owns a slice of the batch (set by parent)
    worker_args = deepcopy(payload["args"])
    worker_args["num_parallel_games"] = int(payload["num_games"])

    # optimizer=None: workers only run selfPlay(), never train()
    agent = AlphaZeroParallel(model, optimizer=None, game=game, args=worker_args)
    return agent.selfPlay()


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    def __repr__(self):
        return "TicTacToe"

    def get_initial_states(self):
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None:
            return False
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]
        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        return encoded_state


class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4

    def __repr__(self):
        return "ConnectFour"

    def get_initial_states(self):
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state

    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None:
            return False
        row = np.min(np.where(state[:, action] != 0))
        player = state[row][action]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0
                    or r >= self.row_count
                    or c < 0
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1
        )

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        return encoded_state


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        # Stored so multiprocessing workers can rebuild an identical net
        self.num_resBlocks = num_resBlocks
        self.num_hidden = num_hidden
        self.device = device

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resBlocks)]
        )
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size),
        )
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh(),
        )
        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        return self.policyHead(x), self.valueHead(x)


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class Node:
    __slots__ = (
        "game", "args", "state", "parent", "action_taken", "prior",
        "children", "visit_count", "value_sum", "C",
    )

    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = float(prior)
        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0.0
        self.C = args["C"]

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -float("inf")
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0.0
        else:
            q_value = 1.0 - ((child.value_sum / child.visit_count) + 1.0) / 2.0
        return q_value + self.C * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def ensure_state(self):
        # Lazy board: only materialize children we actually visit
        if self.state is not None:
            return self.state
        child_state = self.parent.state.copy()
        child_state = self.game.get_next_state(child_state, self.action_taken, 1)
        self.state = self.game.change_perspective(child_state, player=-1)
        return self.state

    def expand(self, policy):
        if hasattr(policy, "tolist"):
            policy = policy.tolist()
        for action, prob in enumerate(policy):
            if prob > 0:
                self.children.append(
                    Node(self.game, self.args, None, self, action, float(prob))
                )

    def backpropagate(self, value):
        value = float(value)
        node = self
        while node is not None:
            node.value_sum += value
            node.visit_count += 1
            value = -value
            node = node.parent

    def clear(self):
        # Break parent↔children cycles for faster GC
        for child in self.children:
            child.parent = None
            child.clear()
        self.children.clear()
        self.parent = None


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.inference_mode()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().tolist()

        noise = np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.action_size)
        eps = self.args["dirichlet_epsilon"]
        policy = [(1 - eps) * p + eps * float(n) for p, n in zip(policy, noise)]

        valid_moves = self.game.get_valid_moves(state)
        policy = [p * float(v) for p, v in zip(policy, valid_moves)]
        s = sum(policy)
        policy = [p / s for p in policy]
        root.expand(policy)

        for _ in range(self.args["num_searches"]):
            node = root
            while node.is_fully_expanded():
                node = node.select()

            node.ensure_state()
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().tolist()
                valid_moves = self.game.get_valid_moves(node.state)
                policy = [p * float(v) for p, v in zip(policy, valid_moves)]
                s = sum(policy)
                policy = [p / s for p in policy]
                value = value.item()
                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    def _infer(self, states):
        encoded = self.game.get_encoded_state(states)
        x = torch.from_numpy(np.ascontiguousarray(encoded)).to(
            self.model.device, dtype=torch.float32, non_blocking=True
        )
        policy, value = self.model(x)
        policy = torch.softmax(policy, dim=1)
        policy_np = policy.detach().cpu().numpy()
        value_np = value.detach().cpu().numpy().reshape(-1)
        return policy_np, value_np

    @torch.inference_mode()
    def search(self, states, spGames):
        policy, _ = self._infer(states)
        eps = self.args["dirichlet_epsilon"]
        noise = np.random.dirichlet(
            [self.args["dirichlet_alpha"]] * self.game.action_size, size=policy.shape[0]
        )
        policy = (1 - eps) * policy + eps * noise

        for i, spg in enumerate(spGames):
            spg_policy = policy[i] * self.game.get_valid_moves(states[i])
            spg_policy /= spg_policy.sum()
            spg.root = Node(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(spg_policy)

        for _ in range(self.args["num_searches"]):
            for spg in spGames:
                spg.node = None
                node = spg.root
                while node.is_fully_expanded():
                    node = node.select()

                node.ensure_state()
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node

            expandable = [i for i, spg in enumerate(spGames) if spg.node is not None]
            if not expandable:
                continue

            batch_states = np.stack([spGames[i].node.state for i in expandable])
            policy, value = self._infer(batch_states)

            for i, idx in enumerate(expandable):
                node = spGames[idx].node
                spg_policy = policy[i] * self.game.get_valid_moves(node.state)
                spg_policy /= spg_policy.sum()
                node.expand(spg_policy)
                node.backpropagate(float(value[i]))


class SPG:
    def __init__(self, game):
        self.state = game.get_initial_states()
        self.memory = []
        self.root = None
        self.node = None


# ---------------------------------------------------------------------------
# AlphaZero
# ---------------------------------------------------------------------------

class AlphaZero:
    """Single-game (non-batched) AlphaZero — used for interactive play demos."""

    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_states()

        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)
            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args["temperature"])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)

            state = self.game.get_next_state(state, action, player)
            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append(
                        (self.game.get_encoded_state(hist_neutral_state), hist_action_probs, hist_outcome)
                    )
                return returnMemory

            player = self.game.get_opponent(player)

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batchIdx:batchIdx + self.args["batch_size"]]
            if not sample:
                continue
            state, policy_targets, value_targets = zip(*sample)
            state = torch.tensor(np.array(state), dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)
            loss = F.cross_entropy(out_policy, policy_targets) + F.mse_loss(out_value, value_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args["num_iterations"]):
            memory = []
            self.model.eval()
            for _ in trange(self.args["num_selfPlay_iterations"]):
                memory += self.selfPlay()
            self.model.train()
            for _ in trange(self.args["num_epochs"]):
                self.train(memory)
            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")


class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

    def selfPlay(self):
        """One batch of parallel games (single process). Also used by workers."""
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for _ in range(self.args["num_parallel_games"])]
        move_count = 0

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)
            self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames) - 1, -1, -1):
                spg = spGames[i]
                action_probs = np.zeros(self.game.action_size, dtype=np.float64)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= action_probs.sum()

                spg.memory.append((spg.root.state, action_probs, player))

                # Explore early, then greedily (shorter games, stronger endplay)
                if move_count < 10:
                    temp = self.args["temperature"]
                    probs = action_probs ** (1 / temp)
                    probs /= probs.sum()
                    action = np.random.choice(self.game.action_size, p=probs)
                else:
                    action = int(np.argmax(action_probs))

                spg.state = self.game.get_next_state(spg.state, action, player)
                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = (
                            value if hist_player == player else self.game.get_opponent_value(value)
                        )
                        return_memory.append(
                            (
                                self.game.get_encoded_state(hist_neutral_state),
                                hist_action_probs,
                                hist_outcome,
                            )
                        )
                    if spg.root is not None:
                        spg.root.clear()
                    del spGames[i]

            player = self.game.get_opponent(player)
            move_count += 1

        return return_memory

    def _selfPlay_multiprocess(self, n_workers: int):
        """
        Split this batch's games across `n_workers` processes.
        Each child calls the same selfPlay() with a smaller num_parallel_games.

        Uses spawn + a top-level worker in THIS module so it is picklable from
        Jupyter (do not define the worker in a notebook cell).
        """
        total_games = self.args["num_parallel_games"]
        game_counts = _partition_counts(total_games, n_workers)
        n_workers = len(game_counts)

        # Weights on CPU for pickling; each worker loads onto its device
        state_dict = {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()}
        device_str = "cuda" if self.model.device.type == "cuda" and torch.cuda.is_available() else "cpu"

        payloads = []
        for worker_id, num_games in enumerate(game_counts):
            payloads.append(
                {
                    "game_name": type(self.game).__name__,
                    "args": self.args,
                    "state_dict": state_dict,
                    "num_resBlocks": self.model.num_resBlocks,
                    "num_hidden": self.model.num_hidden,
                    "device": device_str,
                    "num_games": num_games,
                    # Unique seed per worker / batch for independent exploration
                    "seed": random.randint(0, 2**31 - 1) + worker_id,
                }
            )

        # spawn is required once CUDA has been initialized in the parent.
        # Bind the worker via this module object so children import alphazero_lib,
        # not the notebook's __main__.
        import alphazero_lib as _az_mod

        ctx = mp.get_context("spawn")
        try:
            with ctx.Pool(processes=n_workers) as pool:
                chunks = pool.map(_az_mod.selfplay_worker, payloads)
        except Exception as exc:
            # Fallback keeps training usable if a notebook kernel can't spawn
            print(
                f"WARNING: multiprocess self-play failed ({exc!r}). "
                f"Falling back to 1 process. "
                f"For reliable multi-core, run: python train_alphazero.py"
            )
            return self.selfPlay()

        memory = []
        for chunk in chunks:
            memory.extend(chunk)
        return memory

    def train(self, memory):
        if len(memory) == 0:
            print("WARNING: empty memory — skipping train")
            return
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batchIdx:batchIdx + self.args["batch_size"]]
            if not sample:
                continue
            state, policy_targets, value_targets = zip(*sample)
            state = torch.tensor(np.array(state), dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(
                np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.model.device
            )

            out_policy, out_value = self.model(state)
            loss = F.cross_entropy(out_policy, policy_targets) + F.mse_loss(out_value, value_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        n_batches = self.args["num_selfPlay_iterations"] // self.args["num_parallel_games"]
        if n_batches < 1:
            raise ValueError(
                "num_selfPlay_iterations must be >= num_parallel_games "
                f"(got {self.args['num_selfPlay_iterations']} // {self.args['num_parallel_games']} = 0)"
            )

        # None → auto from os.cpu_count(); int → clamped to available cores
        n_workers = available_selfplay_workers(self.args.get("num_workers"))
        print(
            f"CPU cores={os.cpu_count()} | self-play workers={n_workers} | "
            f"batches/iteration={n_batches} | "
            f"games/batch={self.args['num_parallel_games']} | searches={self.args['num_searches']}"
        )

        for iteration in range(self.args["num_iterations"]):
            memory = []
            self.model.eval()
            gc.disable()
            try:
                for _ in trange(n_batches, desc=f"selfPlay iter {iteration}"):
                    if n_workers == 1:
                        memory += self.selfPlay()
                    else:
                        # Multi-core: same selfPlay(), parallelized across processes
                        memory += self._selfPlay_multiprocess(n_workers)
                    gc.collect()
            finally:
                gc.enable()

            print(f"iteration {iteration}: {len(memory)} training samples")
            self.model.train()
            for _ in trange(self.args["num_epochs"], desc=f"train iter {iteration}"):
                self.train(memory)

            # Optional checkpoint_dir (used by train.py); otherwise cwd
            out_dir = self.args.get("checkpoint_dir") or "."
            os.makedirs(out_dir, exist_ok=True)
            model_path = os.path.join(out_dir, f"model_{iteration}_{self.game}.pt")
            opt_path = os.path.join(out_dir, f"optimizer_{iteration}_{self.game}.pt")
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.optimizer.state_dict(), opt_path)
            print(f"saved {model_path}")
