"""Run under torchrun: verifies every rank ends with identical masks.

Each rank uses a different seed and different data, so without sparsimony's
broadcast the masks would diverge and the ranks would train different subnetworks.

    torchrun --standalone --nproc_per_node=2 ddp_mask_check.py --sparsifier=rigl
"""
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_sparse import get_config  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from sparse_utils import build_sparsifier  # noqa: E402
from sparsimony.utils import get_mask  # noqa: E402


def main():
    from train_sparse import adopt_slurm_env
    adopt_slurm_env()  # so `srun --ntasks=2` works as well as torchrun
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"]) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(local_rank)

    cfg = get_config(sys.argv[1:])
    # Deliberately different per rank: weights, and therefore magnitude-pruning
    # decisions, start out different.
    torch.manual_seed(1234 + rank)

    model = GPT(GPTConfig(block_size=32, vocab_size=128, n_layer=2, n_head=2,
                          n_embd=64, dropout=0.0, bias=False)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sparsifier = build_sparsifier(cfg, model, optimizer, 100)
    ddp_model = DDP(model, device_ids=[local_rank])

    for _ in range(12):
        x = torch.randint(0, 128, (4, 32), device="cuda")
        y = torch.randint(0, 128, (4, 32), device="cuda")
        optimizer.zero_grad()
        _, loss = ddp_model(x, y)
        loss.backward()
        optimizer.step()
        sparsifier.step()

    mismatches = 0
    for group in sparsifier.groups:
        mask = get_mask(group["module"], group["tensor_name"]).to(torch.uint8)
        reference = mask.clone()
        dist.broadcast(reference, 0)
        if not torch.equal(mask, reference):
            print(f"MASK MISMATCH rank {rank}: {group['tensor_fqn']}", flush=True)
            mismatches += 1

    total = torch.tensor([mismatches], device="cuda")
    dist.all_reduce(total)
    if rank == 0:
        print("MASKS_IDENTICAL" if total.item() == 0
              else f"MASKS_DIFFER ({total.item()})", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if total.item() else 0)


if __name__ == "__main__":
    main()
