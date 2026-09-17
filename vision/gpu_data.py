"""GPU-resident data pipeline for the single-GPU vision runs.

The datasets used here are small enough to live on the accelerator in their
entirety (TinyImageNet is 100k x 3 x 64 x 64 uint8 = 1.23 GB train + 0.12 GB
test), and the transforms carry no randomness -- see get_transform in task.py,
which is only ToTensor + Normalize.  That combination means the whole host-side
input pipeline is avoidable: upload the uint8 tensors once, index them with a
permutation generated on device, and normalize in a couple of fused kernels.

This removes per-sample JPEG decode, worker processes, collation and
host-to-device copies from the training loop, which is what was holding GPU
utilization near 10% on TinyImageNet.  Results are unchanged in distribution --
the same samples, the same normalization -- but batch composition follows a
different RNG stream than the DataLoader path, so runs are not bit-comparable
with logs produced before this path existed.
"""

import numpy as np
import torch


def _extract_uint8_chw(dataset):
    """Return the dataset's images as a contiguous uint8 (N, C, H, W) tensor.

    Handles the two in-memory layouts we care about -- task.py's
    TensorImageDataset (already NCHW) and torchvision's CIFAR (NHWC numpy) --
    and returns None for anything else so the caller can fall back to the
    DataLoader path rather than guessing.
    """
    data = getattr(dataset, "data", None)
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if not isinstance(data, torch.Tensor) or data.dtype != torch.uint8:
        return None
    if data.ndim == 3:                                  # (N, H, W) grayscale
        data = data.unsqueeze(1)
    elif data.ndim == 4 and data.shape[-1] in (1, 3):   # (N, H, W, C) -> NCHW
        data = data.permute(0, 3, 1, 2)
    if data.ndim != 4:
        return None
    return data.contiguous()


def _indices_of(dataset, device):
    """Index tensor addressing `dataset`'s samples in the base dataset.

    Chunks are Subsets, but the warm_start benchmark stores the untouched test
    dataset for both levels, which has no .indices.
    """
    indices = getattr(dataset, "indices", None)
    if indices is None:
        return torch.arange(len(dataset), dtype=torch.long, device=device)
    return torch.as_tensor(np.asarray(indices), dtype=torch.long, device=device)


def _stats_tensor(values, device):
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, -1, 1, 1)


class GPUChunkLoader:
    """Iterable yielding batches from GPU-resident tensors.

    Matches the 4-tuple that IndexedDataset produces -- (inputs, labels,
    original_idx, chunk_idx) -- so the training loop does not care which path
    it is fed by.
    """

    def __init__(self, images, targets, indices, chunk_ids, batch_size,
                 mean, std, shuffle=True, generator=None):
        self.images = images
        self.targets = targets
        self.indices = indices
        self.chunk_ids = chunk_ids
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self):
        n = self.indices.numel()
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        device = self.images.device
        n = self.indices.numel()
        if self.shuffle:
            order = torch.randperm(n, device=device, generator=self.generator)
        else:
            order = torch.arange(n, device=device)
        for start in range(0, n, self.batch_size):
            sel = order[start:start + self.batch_size]
            idx = self.indices[sel]
            inputs = self.images[idx].to(torch.float32).div_(255).sub_(self.mean).div_(self.std)
            labels = self.targets[idx]
            chunk_idx = (self.chunk_ids[sel] if self.chunk_ids is not None
                         else torch.full_like(idx, -1))
            yield inputs, labels, idx, chunk_idx


class GPUTaskData:
    """Whole task -- both splits and every chunk's indices -- held on the GPU."""

    def __init__(self, task, device):
        self.device = device

        train_images = _extract_uint8_chw(task._train_dataset)
        test_images = _extract_uint8_chw(task._test_dataset)
        if train_images is None or test_images is None:
            raise TypeError(
                "datasets do not expose uint8 image tensors; "
                "the GPU-resident path needs an in-memory dataset"
            )

        self.train_images = train_images.to(device)
        self.test_images = test_images.to(device)
        self.train_targets = torch.as_tensor(
            np.asarray(task._train_dataset.targets), dtype=torch.long, device=device)
        self.test_targets = torch.as_tensor(
            np.asarray(task._test_dataset.targets), dtype=torch.long, device=device)

        self.train_mean = _stats_tensor(task.train_mean, device)
        self.train_std = _stats_tensor(task.train_std, device)
        self.test_mean = _stats_tensor(task.test_mean, device)
        self.test_std = _stats_tensor(task.test_std, device)

        self.train_indices = [_indices_of(d, device) for d in task._train_datasets]
        self.chunk_ids = [
            torch.as_tensor(np.asarray(d.chunk_indices), dtype=torch.long, device=device)
            if hasattr(d, "chunk_indices") else None
            for d in task._train_datasets
        ]
        self.test_indices = [_indices_of(d, device) for d in task._test_datasets]
        self.full_test_indices = torch.arange(
            len(task._test_dataset), dtype=torch.long, device=device)

    @property
    def nbytes(self):
        return self.train_images.numel() + self.test_images.numel()

    def train_loader(self, level, batch_size, generator=None):
        return GPUChunkLoader(
            self.train_images, self.train_targets,
            self.train_indices[level], self.chunk_ids[level],
            batch_size, self.train_mean, self.train_std,
            shuffle=True, generator=generator,
        )

    @torch.no_grad()
    def evaluate(self, model, level, batch_size, full=False):
        """Top-1 accuracy on one chunk's test split (or the whole test set).

        Accuracy is accumulated on device and synced once at the end rather
        than per batch.
        """
        indices = self.full_test_indices if full else self.test_indices[level]
        n = indices.numel()
        if n == 0:
            return float("nan")

        model.eval()
        correct = torch.zeros((), dtype=torch.long, device=self.device)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            inputs = self.test_images[idx].to(torch.float32).div_(255).sub_(self.test_mean).div_(self.test_std)
            predicted = model(inputs).argmax(dim=1)
            correct += (predicted == self.test_targets[idx]).sum()
        return correct.item() / n


def build_gpu_task_data(task, device):
    """GPUTaskData for `task`, or None when the fast path does not apply."""
    if device.type != "cuda":
        print("[Data] GPU-resident path needs CUDA; using the DataLoader path.")
        return None
    try:
        data = GPUTaskData(task, device)
    except TypeError as exc:
        print(f"[Data] GPU-resident path unavailable ({exc}); using the DataLoader path.")
        return None
    print(f"[Data] GPU-resident: {data.nbytes / 1e9:.2f} GB of uint8 images on {device}")
    return data
