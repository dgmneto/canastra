"""Generate the seeded-random weights fixture behind the `nn-random` bot.

Deliberately tiny arch — the fixture exists so the harness and sandbox can
play *a* network without a training run, not to be good at the game.
"""

from pathlib import Path

from canastra_train import genome

ARCH = {"obs": 2002, "act": 101, "trunk": [32], "head": [16], "activation": "tanh"}
SEED = 20260806
TARGET = Path(__file__).resolve().parents[2] / "bots" / "src" / "fixtures" / "random-init.json"


def main() -> None:
    vec = genome.random_genome(ARCH, seed=SEED)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    genome.save_json(str(TARGET), ARCH, vec)
    print(f"wrote {TARGET} ({genome.genome_size(ARCH)} params)")


if __name__ == "__main__":
    main()