import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from libs.models.SP import MultiScale_GraphConv


def run_case(dataset, node, in_channel):
    B = 2
    T = 32
    V = node
    out_channel = 64

    model = MultiScale_GraphConv(
        num_scales=13,
        in_channels=in_channel,
        out_channels=out_channel,
        dataset=dataset,
        node=node,
        hyper_k=5,
        hyper_hidden=16,
        hyper_dropout=0.0,
        hyper_alpha_init=0.1,
    )

    model.train()

    x = torch.randn(B, in_channel, V, T)
    mask = torch.ones(B, 1, T, dtype=torch.bool)
    joint_graph = torch.eye(V)

    out = model(x, joint_graph, mask=mask)

    expected_shape = (B, out_channel, T, V)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    loss = out.mean()
    loss.backward()

    print(f"[OK] dataset={dataset}, node={node}, input_channel={in_channel}, output_shape={tuple(out.shape)}")


def main():
    run_case("TCG-15", 17, 6)
    run_case("LARA", 19, 12)
    run_case("PKU-subject", 25, 12)
    print("[OK] Adaptive HyperGraph branch smoke test passed.")


if __name__ == "__main__":
    main()
