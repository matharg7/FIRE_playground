"""Plasticity metrics measured on a fixed probe batch.

The metrics follow two public implementations:

  * roger-creus/stable-deep-rl-at-scale (commit 5018636)
      src/utils/representation_dynamics.py  feature ranks, RankMe, feature
                                            norm/mean/std, sharpness
      src/utils/utils.py                    weight norms, gradient norms and
                                            gradient cosine
      src/models/agent.py                   dead units
  * isaac7778/FIRE (commit 3f73d78)
      rl/sac/scale_rl/networks/metrics.py   dormant units (tau = 0.1),
                                            zero-activation units (tau = 0),
                                            effective learning rate
      the FIRE paper                        DfI (eq. 2) and SFE (eq. 1)

adapted to supervised image classification. RL-only metrics (policy churn,
Q-value stability, DR3 co-adaptation) need a policy or a next state and are
not included.

A measurement does not change training:
  * every image comes from a fixed set of training images, chosen once with a
    private random generator (PROBE_SEED), the same in every run;
  * no global random state is consumed: all random draws use private
    generators, and the torch CPU/CUDA random states are also saved and
    restored around each measurement;
  * gradients are computed with torch.autograd.grad, so no .grad buffer is
    written, and no optimizer, scheduler or sparsifier state is touched;
  * the model's train/eval mode is restored and every hook is removed;
  * sparsimony's dense-gradient accumulation, which RigL uses to choose the
    weights it regrows, is frozen and restored (see _SparsifierGuard).
"""

import json

import numpy as np
import torch
import torch.nn as nn

PROBE_SEED = 12345           # fixed, independent of the run's seed
DORMANT_TAU = 0.1            # threshold reported as 'dormant'
DORMANT_TAUS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0)   # also reported
QK_HINTS = ("to_q", "to_k")  # the matrices FIRE changes in a ViT
RANK_TAU = 0.01              # threshold used by the rank estimates

ACT_TYPES = (nn.GELU, nn.ReLU)


def _flatten_acts(a):
    """(rows, units) view of one activation tensor.

    (B, T, H) transformer activations -> (B*T, H): every token is one sample
    of the hidden unit. (B, C, H, W) conv activations -> (B*H*W, C): a channel
    is the unit.
    """
    if a.dim() == 2:
        return a
    if a.dim() == 4:
        return a.permute(0, 2, 3, 1).reshape(-1, a.shape[1])
    return a.reshape(-1, a.shape[-1])


def _weight_matrices(model):
    """(name, 2-D effective weight) for every weight with at least two dims.

    For a sparse layer, module.weight is mask * original, so this is the
    weight the layer actually uses. Conv weights are flattened to
    (out, in * kh * kw).
    """
    out = []
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() < 2:
            continue
        out.append((name, w.detach().reshape(w.shape[0], -1)))
    return out


def _weight_tensors(model):
    """Like _weight_matrices, but conv weights keep their 4-D shape.

    FIRE orthogonalizes a conv weight one kernel slice w[:, :, i, j] at a time,
    so the conv-aware DfI below needs the unflattened tensor.
    """
    out = []
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() < 2:
            continue
        out.append((name, w.detach()))
    return out


class _SparsifierGuard:
    """Keep a measurement out of RigL's regrowth decision.

    Every sparse layer has a backward hook that adds the layer's gradient to
    `dense_grad` while `accumulate` is True. RigL turns accumulation on for
    the step before a mask update and regrows the weights with the largest
    accumulated gradient. The probe runs backward passes, so without this
    guard a measurement made in that window would change which weights RigL
    regrows. The guard turns accumulation off during the measurement and
    restores both `accumulate` and `dense_grad` afterwards.
    """

    def __init__(self, model):
        self.parametrizations = []
        for _, module in model.named_modules():
            if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
                p = module.parametrizations.weight[0]
                if hasattr(p, "accumulate") and hasattr(p, "dense_grad"):
                    self.parametrizations.append(p)
        self.saved = []

    def __enter__(self):
        self.saved = [(p, p.accumulate, p.dense_grad.detach().clone())
                      for p in self.parametrizations]
        for p in self.parametrizations:
            p.accumulate = False
        return self

    def __exit__(self, *exc):
        for p, acc, grad in self.saved:
            p.accumulate = acc
            p.dense_grad = grad
        return False


class PlasticityProbe:
    """Measures plasticity metrics on a fixed batch. One instance per run."""

    def __init__(self, model, task, device, n_images=1024, micro_batch=128,
                 sharpness=False, sharpness_images=256, sharpness_iters=20,
                 sharpness_samples=50, criterion=None):
        self.device = device
        self.micro_batch = micro_batch
        self.sharpness = sharpness
        self.sharpness_images = sharpness_images
        self.sharpness_iters = sharpness_iters
        self.sharpness_samples = sharpness_samples
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.pending = {}           # metrics waiting for the next wandb.log call
        self._prev_grad_dirs = {}   # for the gradient cosine between measurements
        self._prev_features = None  # for the representation drift
        # References for SFE: the weights at initialization and at the start of
        # the current stage (updated by snapshot_stage).
        self._w_init = {n_: w.clone() for n_, w in _weight_matrices(model)}
        self._w_prev = {n_: w.clone() for n_, w in _weight_matrices(model)}

        dataset = task._train_dataset          # full training set
        rs = np.random.RandomState(PROBE_SEED)
        n = min(n_images, len(dataset))
        idx = rs.choice(len(dataset), size=n, replace=False)
        idx.sort()
        images, labels = [], []
        for i in idx:
            x, y = dataset[int(i)][:2]
            images.append(x)
            labels.append(y)
        self.images = torch.stack(images)
        self.labels = torch.tensor(labels)
        self.n_images = n
        # Sum of the chosen indices: equal in two runs iff they probe the same images.
        self.probe_index_checksum = int(idx.sum())

    # ------------------------------------------------------------ activations
    def _activation_sums(self, model):
        """Per-unit mean |a| and mean a for every activation module, and the
        input features of the last Linear layer (the penultimate features).

        A module used more than once in a forward pass (for example one ReLU
        object shared by several layers of an MLP head) is measured separately
        for each use, as '<name>[0]', '<name>[1]', ...; a module used once
        keeps its plain name.
        """
        sums, signed, counts = {}, {}, {}
        calls = {}
        feats = []
        handles = []

        def act_hook(name):
            def hook(_m, _inp, out):
                k = calls.get(name, 0)
                calls[name] = k + 1
                key = (name, k)
                a = _flatten_acts(out.detach().float())
                if key in sums:
                    sums[key] += a.abs().sum(0)
                    signed[key] += a.sum(0)
                    counts[key] += a.shape[0]
                else:
                    sums[key] = a.abs().sum(0)
                    signed[key] = a.sum(0)
                    counts[key] = a.shape[0]
            return hook

        def feat_hook(_m, inp, _out):
            feats.append(_flatten_acts(inp[0].detach().float()).cpu())

        for name, module in model.named_modules():
            if isinstance(module, ACT_TYPES):
                handles.append(module.register_forward_hook(act_hook(name)))
        last_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            handles.append(last_linear.register_forward_hook(feat_hook))

        try:
            with torch.no_grad():
                for s in range(0, self.n_images, self.micro_batch):
                    x = self.images[s:s + self.micro_batch].to(self.device)
                    calls.clear()     # count the uses of each module per forward pass
                    model(x)
        finally:
            for h in handles:
                h.remove()

        uses = {}
        for name, k in sums:
            uses[name] = max(uses.get(name, 0), k + 1)
        acts = {}
        for (name, k) in sums:
            label = name if uses[name] == 1 else f"{name}[{k}]"
            acts[label] = (sums[(name, k)] / counts[(name, k)],
                           signed[(name, k)] / counts[(name, k)])
        features = torch.cat(feats, 0) if feats else None
        return acts, features

    @staticmethod
    def _neuron_metrics(acts):
        """Dormant units for several thresholds, zero-activation and dead units.

        A unit's score is its mean |activation| divided by the mean over its
        layer. It is dormant if the score is <= tau, and a zero-activation unit
        if the score is 0. A unit is dead if its mean signed activation is <= 0;
        after a ReLU that means it is exactly zero on every probe image.
        """
        out = {}
        masks = {t: [] for t in DORMANT_TAUS}
        dead_masks = []
        for name, (mean_abs, mean_signed) in sorted(acts.items()):
            normalized = mean_abs / (mean_abs.mean() + 1e-9)
            for t in DORMANT_TAUS:
                m = (torch.isclose(normalized, torch.zeros_like(normalized)) if t == 0.0
                     else (normalized <= t))
                masks[t].append(m)
                if t == DORMANT_TAU:
                    out[f'dormant/{name}'] = m.float().mean().item() * 100
            dead = (mean_signed <= 0.0)
            dead_masks.append(dead)
            out[f'dead/{name}'] = dead.float().mean().item() * 100
        if dead_masks:
            for t in DORMANT_TAUS:
                key = 'zeroactiv/total' if t == 0.0 else f'dormant_tau{t:g}/total'
                out[key] = torch.cat(masks[t]).float().mean().item() * 100
            out['dormant/total'] = out[f'dormant_tau{DORMANT_TAU:g}/total']
            out['dead/total'] = torch.cat(dead_masks).float().mean().item() * 100
            out['n_neurons'] = int(torch.cat(dead_masks).numel())
        return out

    # ------------------------------------------------------------ DfI and SFE
    @staticmethod
    def _dfi_one(w):
        """(raw, scale_free, per_element) for DfI(W) = ||W^T W - I||_F^2.

        Raw DfI grows with the overall scale of W. The scale-free version first
        rescales W so that ||W||_F^2 = n (singular values with RMS 1), so it
        only measures how far the shape of W is from orthonormal.
        """
        w = w.double()
        n = w.shape[1]
        eye = torch.eye(n, dtype=w.dtype, device=w.device)
        raw = ((w.t() @ w) - eye).pow(2).sum().item()
        fro = w.norm()
        if fro > 0:
            ws = w * (np.sqrt(n) / fro)
            sf = ((ws.t() @ ws) - eye).pow(2).sum().item()
        else:
            sf = float(n)
        return raw, sf, raw / n

    @staticmethod
    def _dfi_slices(w, dfi_one):
        """(raw, scale_free) DfI averaged over the kernel slices of a conv weight.

        For a 2-D weight this is the same as the plain DfI. For a conv weight it
        matches what FIRE does, which is to orthogonalize each w[:, :, i, j].
        """
        if w.dim() == 2:
            raw, sf, _ = dfi_one(w)
            return raw, sf
        raws, sfs = [], []
        for i in range(w.shape[2]):
            for j in range(w.shape[3]):
                r, sf_, _ = dfi_one(w[:, :, i, j])
                raws.append(r)
                sfs.append(sf_)
        return float(np.mean(raws)), float(np.mean(sfs))

    def _dfi_metrics(self, model):
        """DfI per weight and summed / averaged over weights.

        dfi/*        : conv weights flattened to (out, in * kh * kw)
        dfi_slice/*  : conv weights measured per kernel slice (use this for
                       conv networks; the flattened version does not detect
                       FIRE's per-slice orthogonalization)
        dfi_qk/*     : only the to_q / to_k matrices of a ViT
        """
        out = {}
        tot_raw = tot_sf = qk_raw = qk_sf = 0.0
        n_all = n_qk = 0
        for name, w in _weight_matrices(model):
            raw, sf, _ = self._dfi_one(w)
            out[f'dfi/{name}'] = raw
            out[f'dfi_scalefree/{name}'] = sf
            tot_raw += raw
            tot_sf += sf
            n_all += 1
            if any(h in name for h in QK_HINTS):
                qk_raw += raw
                qk_sf += sf
                n_qk += 1
        if n_all:
            out['dfi/total'] = tot_raw
            out['dfi/mean'] = tot_raw / n_all
            out['dfi_scalefree/total'] = tot_sf
            out['dfi_scalefree/mean'] = tot_sf / n_all
            out['dfi/n_matrices'] = n_all
        if n_qk:
            out['dfi_qk/total'] = qk_raw
            out['dfi_qk/mean'] = qk_raw / n_qk
            out['dfi_qk_scalefree/mean'] = qk_sf / n_qk
            out['dfi_qk/n_matrices'] = n_qk

        s_raw = s_sf = 0.0
        n_s = 0
        for name, w in _weight_tensors(model):
            raw_s, sf_s = self._dfi_slices(w, self._dfi_one)
            out[f'dfi_slice/{name}'] = raw_s
            out[f'dfi_slice_scalefree/{name}'] = sf_s
            s_raw += raw_s
            s_sf += sf_s
            n_s += 1
        if n_s:
            out['dfi_slice/total'] = s_raw
            out['dfi_slice/mean'] = s_raw / n_s
            out['dfi_slice_scalefree/total'] = s_sf
            out['dfi_slice_scalefree/mean'] = s_sf / n_s
            out['dfi_slice/n_matrices'] = n_s
        return out

    def _sfe_metrics(self, model):
        """SFE(W, W_ref) = ||W - W_ref||_F^2, against two references.

        sfe_init/*      : the weights at initialization
        sfe_prevstage/* : the weights at the start of the current stage
        'relative' divides by ||W_ref||_F^2, 'per_element' by the number of
        weights.
        """
        out = {}
        for label, ref in (('init', self._w_init), ('prevstage', self._w_prev)):
            if not ref:
                continue
            tot = ref_norm = 0.0
            numel = 0
            for name, w in _weight_matrices(model):
                if name not in ref or ref[name].shape != w.shape:
                    continue
                tot += (w.double() - ref[name].double()).pow(2).sum().item()
                ref_norm += ref[name].double().pow(2).sum().item()
                numel += w.numel()
            if numel:
                out[f'sfe_{label}/raw'] = tot
                out[f'sfe_{label}/relative'] = tot / ref_norm if ref_norm > 0 else 0.0
                out[f'sfe_{label}/per_element'] = tot / numel
        return out

    def snapshot_stage(self, model):
        """Store the current weights as the reference for sfe_prevstage."""
        self._w_prev = {n_: w.clone() for n_, w in _weight_matrices(model)}

    # ------------------------------------------------------------ features
    @staticmethod
    def _rank_metrics(features):
        """Rank estimates and size statistics of the penultimate features (N, D)."""
        f = features.double()
        svals = torch.linalg.svdvals(f)
        n_obs = f.shape[0]
        out = {}

        # effective rank (Roy & Vetterli, 2007)
        sval_sum = svals.sum()
        dist = svals / sval_sum
        dist_fixed = torch.where(dist == 0, torch.ones_like(dist), dist)
        out['rank/effective_vetterli'] = torch.exp(-(dist_fixed * dist_fixed.log()).sum()).item()

        # approximate rank: directions needed for 99% of the variance (Yang et al., 2020)
        sq = svals ** 2
        crossed = torch.cumsum(sq, 0) >= (1 - RANK_TAU) * sq.sum()
        out['rank/approximate_pca'] = int((~crossed).sum().item()) + 1

        # srank (Kumar et al., 2020)
        crossed = torch.cumsum(svals, 0) >= (1 - RANK_TAU) * sval_sum
        out['rank/srank_kumar'] = int((~crossed).sum().item()) + 1

        # feature rank (Lyle et al., 2022)
        out['rank/feature_lyle'] = int((svals / np.sqrt(n_obs) > RANK_TAU).sum().item())

        # numerical matrix rank
        out['rank/matrix'] = int(torch.linalg.matrix_rank(f).item())

        # RankMe as computed in stable-deep-rl-at-scale: the entropy of the
        # covariance eigenvalues (squared singular values)
        p = sq / sq.sum() + 1e-6
        out['rank/rankme_cov'] = torch.exp(-(p * p.log()).sum()).item()

        out['feature/norm'] = torch.linalg.norm(f, ord=2, dim=1).mean().item()
        out['feature/mean'] = f.mean(dim=-1).mean().item()
        out['feature/std'] = f.std(dim=-1).mean().item()
        out['feature/sval_1'] = (svals[0] / np.sqrt(n_obs)).item()
        out['feature/sval_last'] = (svals[-1] / np.sqrt(n_obs)).item()
        out['feature/dim'] = int(f.shape[1])
        return out

    def _drift_metrics(self, features):
        """How far the penultimate features moved since the previous measurement."""
        out = {}
        if self._prev_features is not None and self._prev_features.shape == features.shape:
            a, b = self._prev_features.double(), features.double()
            out['drift/cosine'] = torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()
            out['drift/l2'] = torch.nn.functional.pairwise_distance(a, b).mean().item()
        self._prev_features = features.clone()
        return out

    # ------------------------------------------------------------ weights
    @staticmethod
    def _weight_metrics(model):
        """Weight norms of the effective weights, and for sparse layers the
        share of units whose incoming or outgoing weights are all masked."""
        out = {}
        total_sq = 0.0
        n_weights = 0
        dead_out = dead_in = n_out = n_in = 0
        sparse_modules = 0
        for name, module in model.named_modules():
            w = getattr(module, 'weight', None)
            if not isinstance(w, torch.Tensor) or w.dim() < 2:
                continue
            with torch.no_grad():
                eff = w.detach()            # mask * original for a sparse layer
                out[f'wnorm/{name}'] = eff.norm().item()
                total_sq += (eff ** 2).sum().item()
                n_weights += eff.numel()
                parametrized = (hasattr(module, 'parametrizations')
                                and 'weight' in module.parametrizations)
                if parametrized:
                    sparse_modules += 1
                    live = (eff != 0)
                    flat = live.reshape(live.shape[0], -1)
                    dead_out += int((~flat.any(dim=1)).sum().item())
                    n_out += flat.shape[0]
                    flat_in = live.transpose(0, 1).reshape(live.shape[1], -1)
                    dead_in += int((~flat_in.any(dim=1)).sum().item())
                    n_in += flat_in.shape[0]
        out['wnorm/total'] = float(np.sqrt(total_sq))
        out['wnorm/n_weights'] = n_weights
        if sparse_modules:
            out['sparse/fully_masked_out_units_pct'] = dead_out / max(n_out, 1) * 100
            out['sparse/fully_masked_in_units_pct'] = dead_in / max(n_in, 1) * 100
            out['sparse/n_sparse_modules'] = sparse_modules
        return out

    # ------------------------------------------------------------ gradients
    def _grad_metrics(self, model):
        """Gradient norm, effective learning rate (gradient norm / parameter
        norm) and the cosine between this gradient and the previous one, for
        every trainable parameter with at least two dims. For a sparse layer
        the parameter is the stored (unmasked) tensor."""
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
        if not params:
            return {}
        names = [n for n, _ in params]
        tensors = [p for _, p in params]
        total_sq = 0.0
        grads = [torch.zeros_like(p) for p in tensors]
        n_rows = 0
        with torch.enable_grad():
            for s in range(0, self.n_images, self.micro_batch):
                x = self.images[s:s + self.micro_batch].to(self.device)
                y = self.labels[s:s + self.micro_batch].to(self.device)
                loss = self.criterion(model(x), y)
                g = torch.autograd.grad(loss, tensors, allow_unused=True)
                for i, gi in enumerate(g):
                    if gi is not None:
                        grads[i] += gi.detach() * x.shape[0]
                n_rows += x.shape[0]
        out = {}
        for name, p, g in zip(names, tensors, grads):
            g = g / max(n_rows, 1)
            gn = g.norm().item()
            pn = p.detach().norm().item()
            total_sq += gn ** 2
            out[f'gnorm/{name}'] = gn
            out[f'effective_lr/{name}'] = gn / pn if pn > 0 else 0.0
            unit = (g / gn).flatten() if gn > 0 else g.flatten()
            prev = self._prev_grad_dirs.get(name)
            if prev is not None and prev.shape == unit.shape:
                out[f'grad_cosine/{name}'] = torch.dot(unit, prev).item()
            self._prev_grad_dirs[name] = unit.clone()
        out['gnorm/total'] = float(np.sqrt(total_sq))
        cosines = [v for k, v in out.items() if k.startswith('grad_cosine/')]
        if cosines:
            out['grad_cosine/mean'] = float(np.mean(cosines))
        return out

    # ------------------------------------------------------------ sharpness
    def _sharpness_metrics(self, model):
        """Largest Hessian eigenvalue (power iteration) and Hutchinson estimate
        of the Hessian trace, on the first sharpness_images probe images."""
        n = min(self.sharpness_images, self.n_images)
        x = self.images[:n].to(self.device)
        y = self.labels[:n].to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
        with torch.enable_grad():
            loss = self.criterion(model(x), y)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            flat_grad = torch.cat([g.reshape(-1) for g in grads])

            def hvp(v):
                hv = torch.autograd.grad(torch.dot(flat_grad, v), params, retain_graph=True)
                return torch.cat([h.reshape(-1) for h in hv]).detach()

            v = torch.randn(flat_grad.numel(), device=flat_grad.device,
                            generator=torch.Generator(device=flat_grad.device).manual_seed(PROBE_SEED))
            v /= v.norm()
            for _ in range(self.sharpness_iters):
                hv = hvp(v)
                nrm = hv.norm()
                if nrm == 0:
                    break
                v = hv / nrm
            lam = torch.dot(v, hvp(v)).item()

            gen = torch.Generator(device=flat_grad.device).manual_seed(PROBE_SEED + 1)
            trace = 0.0
            for _ in range(self.sharpness_samples):
                z = torch.randint(0, 2, (flat_grad.numel(),), device=flat_grad.device,
                                  generator=gen, dtype=torch.float32) * 2 - 1
                trace += torch.dot(z, hvp(z)).item()
            trace /= max(self.sharpness_samples, 1)
        return {'sharpness/lambda_max': lam, 'sharpness/hessian_trace': trace}

    # ------------------------------------------------------------ measure
    def measure(self, model, stage, epoch, tag, sharpness=None):
        """Compute every metric and return them as 'plast/<tag>/<metric>'.

        Each measurement is also printed as one '[plast] {json}' line and
        queued in self.pending for the next wandb.log call. `sharpness`
        overrides the constructor's default for this call, so the Hessian
        metrics (about 20 extra backward passes) can be limited to stage
        boundaries.
        """
        was_training = model.training
        model.eval()
        do_sharp = self.sharpness if sharpness is None else sharpness
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            with _SparsifierGuard(model):
                acts, features = self._activation_sums(model)
                m = {}
                m.update(self._neuron_metrics(acts))
                if features is not None:
                    m.update(self._rank_metrics(features))
                    m.update(self._drift_metrics(features))
                m.update(self._weight_metrics(model))
                m.update(self._dfi_metrics(model))
                m.update(self._sfe_metrics(model))
                m.update(self._grad_metrics(model))
                if do_sharp:
                    m.update(self._sharpness_metrics(model))
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            if was_training:
                model.train()

        m['stage'] = stage
        m['epoch'] = epoch
        m['tag'] = tag
        print('[plast] ' + json.dumps(m, sort_keys=True), flush=True)
        out = {f'plast/{tag}/{k}': v for k, v in m.items() if k != 'tag'}
        self.pending.update(out)
        return out
