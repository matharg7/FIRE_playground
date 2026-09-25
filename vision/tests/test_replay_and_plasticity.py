"""Tests for replay_ratio and the plasticity probe.

Everything runs on CPU with tiny synthetic datasets and models.
"""

import types

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# replay_ratio
# ---------------------------------------------------------------------------

def _tiny_task_class():
    from task import Task, TensorImageDataset

    class TinyTask(Task):
        """8 classes, 20 training and 5 test images per class."""
        n_classes, n_train, n_test = 8, 20, 5

        def _get_dataset(self):
            g = torch.Generator().manual_seed(0)
            def split(n):
                data = torch.randint(0, 256, (self.n_classes * n, 3, 4, 4),
                                     dtype=torch.uint8, generator=g)
                targets = np.repeat(np.arange(self.n_classes), n)
                return TensorImageDataset(data, targets, list(range(self.n_classes)),
                                          ((0.5,) * 3, (0.25,) * 3))
            return split(self.n_train), split(self.n_test)

        @property
        def shape(self):
            return [3, 4, 4]

    return TinyTask


def _sample_task(r, seed=0):
    return _tiny_task_class()(mode='sample', n_chunks=4, access='full',
                              test_access='same', seed=seed, replay_ratio=r)


def _class_task(r, seed=0):
    return _tiny_task_class()(mode='class', n_chunks=4, access='full',
                              test_access='same', seed=seed, replay_ratio=r)


def _indices(task):
    return [set(int(i) for i in d.indices) for d in task._train_datasets]


def test_replay_ratio_one_keeps_the_cumulative_chunks():
    task = _sample_task(1.0)
    chunk = len(task._train_dataset) // 4
    for i, d in enumerate(task._train_datasets):
        assert len(d) == (i + 1) * chunk


@pytest.mark.parametrize("r", [0.5, 0.25])
def test_sample_incremental_replay(r):
    task = _sample_task(r)
    stages = _indices(task)
    n_chunk = len(task._train_dataset) // 4
    new = [stages[0]] + [stages[i] - stages[i - 1] for i in range(1, 4)]
    for i in range(1, 4):
        # Chunk i is new at stage i and fully included.
        assert len(new[i]) >= n_chunk
        # Stage i = all of its own chunk + round(r * chunk) of each earlier chunk.
        assert len(stages[i]) == n_chunk + i * int(round(r * n_chunk))
    # The part of chunk 0 replayed at stage 1 is exactly the part replayed later.
    kept0 = stages[1] & stages[0]
    assert len(kept0) == int(round(r * n_chunk))
    for i in range(2, 4):
        assert stages[i] & stages[0] == kept0
    # Another Task instance with the same seed replays the same images.
    assert _indices(_sample_task(r)) == stages


@pytest.mark.parametrize("r", [0.5, 0.2])
def test_class_incremental_replay(r):
    full, task = _class_task(1.0), _class_task(r)
    targets = task._train_dataset.targets
    per_class = task.n_train
    seen_before = set()
    kept = {}
    for i, (d, d_full) in enumerate(zip(task._train_datasets, full._train_datasets)):
        idx = np.asarray(d.indices)
        classes_now = set(int(c) for c in np.unique(targets[np.asarray(d_full.indices)]))
        new_classes = classes_now - seen_before
        for c in classes_now:
            members = set(int(j) for j in idx[targets[idx] == c])
            if c in new_classes:
                assert len(members) == per_class              # new class: all images
            else:
                assert len(members) == int(round(r * per_class))
                kept.setdefault(c, members)
                assert members == kept[c]                    # same subset at every stage
        assert set(int(c) for c in np.unique(targets[idx])) == classes_now
        # The test set is never reduced.
        assert list(task._test_datasets[i].indices) == list(full._test_datasets[i].indices)
        seen_before = classes_now


@pytest.mark.parametrize("r", [0.0, 1.5, -0.1])
def test_replay_ratio_out_of_range_raises(r):
    with pytest.raises(ValueError):
        _sample_task(r)


def test_replay_ratio_not_defined_for_warm_start():
    with pytest.raises(ValueError):
        _tiny_task_class()(mode='sample', n_chunks=2, access='full',
                           test_access='same', replay_ratio=0.5)


# ---------------------------------------------------------------------------
# plasticity probe
# ---------------------------------------------------------------------------

class SharedReluNet(nn.Module):
    """One ReLU object used after two layers, as in the VGG16 MLP head."""

    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(6, 8)
        self.fc1 = nn.Linear(8, 8)
        self.out = nn.Linear(8, 3)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.out(self.act(self.fc1(self.act(self.fc0(x)))))


class _Data(torch.utils.data.Dataset):
    def __init__(self, n=40):
        g = torch.Generator().manual_seed(1)
        self.x = torch.randn(n, 6, generator=g)
        self.y = torch.randint(0, 3, (n,), generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def _probe(model, **kw):
    from interventions.plasticity_metrics import PlasticityProbe
    task = types.SimpleNamespace(_train_dataset=_Data())
    kw = {'n_images': 32, 'micro_batch': 8, **kw}
    return PlasticityProbe(model, task, 'cpu', **kw)


def test_probe_uses_the_same_images_every_time():
    a, b = _probe(SharedReluNet()), _probe(SharedReluNet())
    assert a.probe_index_checksum == b.probe_index_checksum
    assert torch.equal(a.images, b.images) and torch.equal(a.labels, b.labels)


def test_shared_activation_is_measured_per_use():
    torch.manual_seed(0)
    model = SharedReluNet()
    with torch.no_grad():                  # make unit 0 of fc0 dead
        model.fc0.weight[0].zero_()
        model.fc0.bias[0] = -1.0
    probe = _probe(model)
    m = probe.measure(model, stage=0, epoch=0, tag='epoch')
    assert 'plast/epoch/dead/act' not in m
    with torch.no_grad():
        a0 = model.act(model.fc0(probe.images))
        a1 = model.act(model.fc1(a0))
    for k, a in enumerate((a0, a1)):
        dead = (a.mean(0) <= 0).float().mean().item() * 100
        assert m[f'plast/epoch/dead/act[{k}]'] == pytest.approx(dead)
        score = a.abs().mean(0) / (a.abs().mean(0).mean() + 1e-9)
        dormant = (score <= 0.1).float().mean().item() * 100
        assert m[f'plast/epoch/dormant/act[{k}]'] == pytest.approx(dormant)
    assert m['plast/epoch/dead/act[0]'] >= 100 / 8
    assert m['plast/epoch/n_neurons'] == 16


def test_measurement_does_not_change_training():
    from config_st import Config
    from train_st import build_sparsifier
    torch.manual_seed(0)
    model = SharedReluNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    cfg = Config({'sparsifier': 'rigl', 'sparsity': 0.5, 'num_mask_updates': 5,
                  't_end_ratio': 0.8, 'pruning_ratio': 0.3,
                  'drop_fraction_schedule': 'global', 't_accel_ratio': 0.2,
                  'initial_sparsity': 0.0, 'grow_init': 'zero'})
    build_sparsifier(cfg, model, opt, chunk_steps=[20])
    probe = _probe(model, sharpness=True, sharpness_images=8,
                   sharpness_iters=3, sharpness_samples=3)

    x, y = torch.randn(8, 6), torch.randint(0, 3, (8,))
    nn.CrossEntropyLoss()(model(x), y).backward()     # leave .grad buffers set
    sparse = [m.parametrizations.weight[0] for m in model.modules()
              if hasattr(m, 'parametrizations')]
    for p in sparse:                                   # as in the step before a RigL update
        p.accumulate = True
        p.dense_grad = torch.randn_like(p.dense_grad)
    model.train()

    state = {k: v.clone() for k, v in model.state_dict().items()}
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    guard = [(p.accumulate, p.dense_grad.clone()) for p in sparse]
    torch_rng, np_rng = torch.get_rng_state(), np.random.get_state()

    probe.measure(model, stage=0, epoch=0, tag='epoch', sharpness=True)

    assert model.training
    for k, v in model.state_dict().items():
        assert torch.equal(v, state[k]), k
    for n, p in model.named_parameters():
        if n in grads:
            assert torch.equal(p.grad, grads[n]), n
    for p, (acc, dg) in zip(sparse, guard):
        assert p.accumulate == acc and torch.equal(p.dense_grad, dg)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert np.array_equal(np.random.get_state()[1], np_rng[1])


def test_dfi_known_values():
    from interventions.plasticity_metrics import PlasticityProbe
    q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
    raw, sf, _ = PlasticityProbe._dfi_one(q)
    assert raw == pytest.approx(0, abs=1e-10) and sf == pytest.approx(0, abs=1e-10)
    w = torch.randn(8, 5, dtype=torch.float64)
    assert PlasticityProbe._dfi_one(3 * w)[1] == pytest.approx(PlasticityProbe._dfi_one(w)[1])
    conv = torch.stack([torch.stack([q, q], -1), torch.stack([q, q], -1)], -1)   # (8, 8, 2, 2)
    assert PlasticityProbe._dfi_slices(conv, PlasticityProbe._dfi_one)[1] == pytest.approx(0, abs=1e-10)


def test_rank_of_rank_three_features():
    from interventions.plasticity_metrics import PlasticityProbe
    g = torch.Generator().manual_seed(0)
    f = (torch.randn(100, 3, generator=g, dtype=torch.float64)
         @ torch.randn(3, 8, generator=g, dtype=torch.float64))
    assert PlasticityProbe._rank_metrics(f)['rank/matrix'] == 3
    assert PlasticityProbe._rank_metrics(f)['rank/srank_kumar'] <= 3


def test_sfe_measures_the_weight_change():
    torch.manual_seed(0)
    model = SharedReluNet()
    probe = _probe(model)
    assert probe._sfe_metrics(model)['sfe_init/raw'] == 0.0
    with torch.no_grad():
        model.fc1.weight[0, 0] += 0.5
    assert probe._sfe_metrics(model)['sfe_init/raw'] == pytest.approx(0.25)
    probe.snapshot_stage(model)
    m = probe._sfe_metrics(model)
    assert m['sfe_prevstage/raw'] == 0.0 and m['sfe_init/raw'] == pytest.approx(0.25)
