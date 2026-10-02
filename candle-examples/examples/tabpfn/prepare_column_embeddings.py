"""Reproduce column_embeddings.safetensors using the pinned upstream checkout.

Usage: python prepare_column_embeddings.py /path/to/TabPFN
Requires torch==2.8.0 and safetensors. The upstream source commit is
de50fac665bdfacba41b049aa3a942c70e7c5001.
"""

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors.torch import save_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("upstream", type=Path)
args = parser.parse_args()
source = args.upstream / "src/tabpfn/architectures/shared/tabpfn_col_embedding.pt"
assert torch.__version__.split("+")[0] == "2.8.0", "use PyTorch 2.8.0 to reproduce the asset"
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    "50802de6718984f275ae67b10ecfe93dfeb75e5804bb579294bd77053cb95b32"
), "column embeddings do not match the pinned upstream source"
v2 = torch.randn((2000, 48), generator=torch.Generator().manual_seed(0))
# The CPU vectorized normal generator has identical prefixes for these sizes.
for groups in (1, 2, 3, 17, 2000):
    expected = torch.randn((groups, 48), generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(v2[:groups], expected, rtol=0, atol=0)
v2_5 = torch.load(source, map_location="cpu", weights_only=True)
assert v2_5.shape == (2000, 48) and v2_5.dtype == torch.float32
save_file(
    {"v2": v2, "v2_5": v2_5},
    str(Path(__file__).with_name("column_embeddings.safetensors")),
    metadata={
        "upstream_commit": "de50fac665bdfacba41b049aa3a942c70e7c5001",
        "v2": "torch 2.8.0 CPU randn((2000,48)), seed=0",
        "v2_5": "upstream tabpfn_col_embedding.pt, also used by v2.6",
    },
)
