# AlphaZero From Scratch

Connect Four AlphaZero trained end-to-end: game env → MCTS → ResNet (policy + value) → self-play → multi-core parallel training.

Built by working through [foersterrobert/AlphaZeroFromScratch](https://github.com/foersterrobert/AlphaZeroFromScratch), with performance fixes for a real multi-core + GPU machine.

## Layout

```
AlphaZero_From_Scratch/
├── alphazero_lib.py      # games, ResNet, MCTS, AlphaZeroParallel, workers
├── train.py              # train Connect Four (saves into checkpoints/)
├── play.py               # human vs trained model
├── requirements.txt
├── checkpoints/
│   ├── model_7_ConnectFour.pt      # final network weights
│   └── optimizer_7_ConnectFour.pt  # final Adam state
└── README.md
```

## Setup

```bash
cd AlphaZero_From_Scratch
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CUDA optional — training and play fall back to CPU if no GPU is present.

## Play the trained model

```bash
python play.py
```

You are player `1`. Enter a column `0–6`. The AI uses `checkpoints/model_7_ConnectFour.pt`.

## Train from scratch

```bash
python train.py
```

Default run (what produced `model_7`):

| Setting | Value |
|---|---|
| Network | ResNet, **9** residual blocks × **128** channels |
| Game | Connect Four (6×7) |
| MCTS searches / move | 600 |
| Iterations | 8 |
| Games / iteration | 1024 (`num_selfPlay_iterations`) |
| Parallel games / batch | 256 |
| Self-play workers | `os.cpu_count() - 1` (auto) |
| Train batch size | 2048 |
| Epochs / iteration | 4 |
| Temperature | 1.25 |
| Dirichlet | ε=0.25, α=0.3 |

On an RTX 3090 + 12-thread CPU this was about **~1.5–2 hours** wall time for the full 8 iterations.

Override workers if needed in `train.py`:

```python
"num_workers": None,  # auto
# "num_workers": 4,  # fixed
```

## What the code does

1. **`ConnectFour` / `TicTacToe`** — board rules, valid moves, win/draw, encoding for the net  
2. **`ResNet`** — shared trunk + policy head + value head (`tanh`)  
3. **`MCTS` / `MCTSParallel`** — PUCT search; neural net replaces rollouts at expand/eval  
4. **`AlphaZeroParallel.selfPlay`** — many games in one batch, visit counts → policy targets, game outcome → value targets  
5. **`learn`** — self-play → train → save checkpoint each iteration  

Self-play workers call the **same** `selfPlay()` (no duplicated logic). `spawn` is used so CUDA stays safe after the parent has initialized the GPU.

## Optimizations (vs the basic tutorial loop)

These mattered more than chasing GPU util %:

| Change | Why |
|---|---|
| **Multi-process self-play** | Tutorial MCTS is mostly single-thread Python. Workers = `cpu_count - 1`, games split evenly. |
| **Larger `num_parallel_games` (256)** | Bigger batched NN forwards during search. |
| **Larger train `batch_size` (2048)** | Better GPU use in the short training phase. |
| **Lazy child boards** | Expand stores priors; board materializes on first visit. |
| **Python floats in the hot path** | Avoids NumPy scalar overhead in UCB / backprop. |
| **`Node.__slots__` + iterative backprop** | Less allocation / recursion. |
| **Break tree cycles + GC around self-play** | Parent↔children cycles were keeping millions of nodes alive. |
| **Greedy moves after ply 10** | Shorter games, less random endplay noise. |
| **TF32 / cudnn.benchmark** | Small win on Ampere GPUs during train/infer. |

**Honest note:** on Connect Four the GPU still will not sit at 100% during self-play — the tree walk is CPU-bound. Measure **games/hour**, not only `nvidia-smi`.

Observed on this project: first self-play batch ~**1 hour → under 3 minutes** after multi-core + the MCTS path fixes; full 8-iteration train ~**1.5–2 hours**.

## Checkpoints

| File | Description |
|---|---|
| `checkpoints/model_7_ConnectFour.pt` | Weights after iteration 7 (final) |
| `checkpoints/optimizer_7_ConnectFour.pt` | Adam state after iteration 7 |

Architecture must match when loading: `ResNet(game, 9, 128, device)`.

## Credits

- Tutorial / reference notebooks: [foersterrobert/AlphaZeroFromScratch](https://github.com/foersterrobert/AlphaZeroFromScratch)  
- AlphaZero paper: [arXiv:1712.01815](https://arxiv.org/abs/1712.01815)
