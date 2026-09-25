"""Weight Space Consolidation (WSC) for continual fine-tuning.

Ported from `models/wsc.py` of umamicode/weight-space-consolidation
(commit 026c54e9, Apache-2.0), the reference implementation for
"Forget Forgetting: Continual Learning in a World of Abundant Memory"
(Cho et al., ICLR 2026, arXiv:2502.07274).

WSC adds two operations at each task boundary, on top of replay:

  1. rank-based parameter reset ("trim"): score every weight by |m_hat| * v_hat,
     the bias-corrected Adam moments of its gradient accumulated over the task.
     Keep the top `retain_percent` of each tensor untouched and blend the rest
     half-way back to the previous task's weights. Restores plasticity.
  2. stochastic weight averaging: once validation loss plateaus, trim once, then
     average weights over the remaining epochs. Adds stability.

NOT ported from the upstream repo's `trace/wsc_llm.py`: that module is JAX, and
the PyTorch entry points its own INTEGRATION.md tells you to import
(`make_swa_torch`, `make_swa_deepspeed`) raise NotImplementedError. The
documented names (`clear_moment_score_state`, `snapshot_trainable_params_cpu`)
do not exist in it either. The working reference is the vision `models/wsc.py`,
which is what this file follows.

Three adaptations were required, all recorded in PLAN.md:

  * `swa_lr`. Upstream uses 0.1, tuned for SGD on CIFAR. At our 1e-5 that would
    destroy the model, so swa_lr defaults to the run's own learning_rate.
  * `patience`. Upstream uses 4 epochs over a ~170-epoch vision schedule. TRACE
    gives these tasks 3-5 epochs, where 4 can never fire. check_wsc_schedule()
    refuses to start a run whose tasks are too short to trigger SWA at all.
  * BatchNorm recalibration is dropped: `update_bn` is a no-op on a transformer
    with no BatchNorm layers.
"""

import copy

import torch


# --------------------------------------------------------------------------
# importance scoring
# --------------------------------------------------------------------------

class MomentTracker:
    """EMA of the gradient's first and second moments, per parameter.

    Upstream accumulates these with Adam's betas during the task and scores
    importance as |m_hat| * v_hat after bias correction. It is deliberately a
    separate accumulator from the optimizer's own state: AdamW's moments are
    reset at each task boundary (--reset_optimizer), while these must cover the
    task that just finished.
    """

    def __init__(self, model, beta1=0.9, beta2=0.999):
        self.beta1, self.beta2 = beta1, beta2
        self.t = 0
        self._m = {}
        self._v = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._m[name] = torch.zeros_like(p, memory_format=torch.preserve_format)
                self._v[name] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def update(self, model):
        """Call once per optimizer step, after backward and before zero_grad."""
        self.t += 1
        for name, p in model.named_parameters():
            if p.grad is None or name not in self._m:
                continue
            g = p.grad
            self._m[name].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            self._v[name].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)

    @torch.no_grad()
    def scores(self):
        """|m_hat| * v_hat per parameter, bias-corrected exactly as Adam does."""
        if self.t == 0:
            return {}
        c1 = 1 - self.beta1 ** self.t
        c2 = 1 - self.beta2 ** self.t
        return {n: (self._m[n] / c1).abs() * (self._v[n] / c2) for n in self._m}


@torch.no_grad()
def snapshot_params(model):
    """theta(t-1): a CPU copy of the trainable weights, for the trim to blend to."""
    return {n: p.detach().to("cpu", copy=True)
            for n, p in model.named_parameters() if p.requires_grad}


# --------------------------------------------------------------------------
# rank-based parameter reset
# --------------------------------------------------------------------------

@torch.no_grad()
def pre_swa_trim(model, prev_state, scores, retain_percent=20.0):
    """Blend low-importance weights half-way back toward the previous task.

    Per tensor, keep the top `retain_percent`% by score and set the rest to
    0.5 * current + 0.5 * previous. The threshold is per-parameter, not global,
    matching upstream (which flattens all scores but then slices and takes topk
    within each tensor).

    Returns (tensors_trimmed, weights_blended).
    """
    if not prev_state or not scores:
        return 0, 0
    keep = retain_percent / 100.0
    n_tensors = n_blended = 0
    for name, p in model.named_parameters():
        old = prev_state.get(name)
        s = scores.get(name)
        if old is None or s is None or old.shape != p.shape:
            continue
        n = p.numel()
        k = max(1, int(n * keep))
        thresh = s.flatten().topk(k, largest=True).values.min()
        mask = (s >= thresh)                       # True = keep untouched
        old_dev = old.to(p.device, dtype=p.dtype)
        blended = p.data.mul(0.5).add_(old_dev, alpha=0.5)
        p.data = torch.where(mask, p.data, blended)
        n_tensors += 1
        n_blended += int((~mask).sum().item())
    return n_tensors, n_blended


# --------------------------------------------------------------------------
# plateau detection
# --------------------------------------------------------------------------

class PlateauWatcher:
    """Fires once validation loss has failed to improve for `patience` epochs."""

    def __init__(self, patience=4, tol=1e-4):
        self.patience, self.tol = patience, tol
        self.best = float("inf")
        self.stagnant = 0
        self.fired = False

    def step(self, val_loss):
        if val_loss + self.tol < self.best:
            self.best = val_loss
            self.stagnant = 0
        else:
            self.stagnant += 1
        if not self.fired and self.stagnant >= self.patience:
            self.fired = True
            return True
        return False


def check_wsc_schedule(cfg, epochs_list, task_names=None):
    """Refuse to start a run whose tasks are too short for SWA to ever fire.

    Upstream's patience of 4 sits inside a ~170-epoch vision schedule. TRACE
    gives these tasks 3-5 epochs: plateau detection needs `patience`
    CONSECUTIVE non-improving epochs, and the first epoch can only set the
    baseline, so a task needs at least patience+2 epochs for SWA to run for even
    one epoch. Otherwise WSC silently degenerates into plain fine-tuning.
    """
    if cfg.cl_method != 'wsc':
        return
    need = cfg.wsc_patience + 2
    names = task_names or [f"task {i}" for i in range(len(epochs_list))]
    short = [(n, int(e)) for n, e in zip(names, epochs_list) if int(e) < need]
    print("[wsc] epochs per task: "
          + ", ".join(f"{n}={int(e)}" for n, e in zip(names, epochs_list))
          + f"  (need >= {need} for SWA to run, patience={cfg.wsc_patience})")
    if short:
        raise ValueError(
            "WSC cannot trigger on " + ", ".join(f"{n} ({e} epochs)" for n, e in short)
            + f": plateau detection needs {cfg.wsc_patience} consecutive "
              f"non-improving epochs, so a task needs >= {need} epochs. "
              f"Either raise --epochs_per_task or lower --wsc_patience "
              f"(upstream's 4 assumes a ~170-epoch vision schedule).")


# --------------------------------------------------------------------------
# stochastic weight averaging
# --------------------------------------------------------------------------

def make_swa(model, optimizer, swa_lr, anneal_epochs=5):
    """AveragedModel plus a cosine-annealed constant SWA learning rate.

    swa_lr must be on the scale of the run's own LR: upstream's 0.1 is an SGD
    vision value and would wreck a 1e-5 fine-tune. update_bn is not needed --
    a transformer has no BatchNorm.
    """
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_sched = torch.optim.swa_utils.SWALR(
        optimizer, anneal_strategy="cos", anneal_epochs=anneal_epochs, swa_lr=swa_lr)
    return swa_model, swa_sched


@torch.no_grad()
def load_swa_into(model, swa_model):
    """Copy the averaged weights back into the live model."""
    src = swa_model.module.state_dict()
    model.load_state_dict({k: v for k, v in src.items()}, strict=True)


# --------------------------------------------------------------------------
# per-task controller
# --------------------------------------------------------------------------

class WSCController:
    """Owns WSC's per-task state and the hooks train_task calls.

    Lifecycle, mirroring upstream `_update_representation`:
        begin_task(t)         snapshot theta(t-1); reset the moment tracker
        on_optimizer_step()   accumulate |g| and g^2 EMAs
        on_epoch_end()        val loss -> plateau check -> trim + start SWA;
                              while SWA is active, update the average and step SWALR
        end_task()            load the averaged weights back into the model

    Task 0 does plain fine-tuning: there is no previous task to blend toward,
    exactly as upstream routes task 0 through `_init_train`.
    """

    def __init__(self, cfg, model, val_loss_fn, log=print):
        self.cfg = cfg
        self.val_loss_fn = val_loss_fn        # () -> float, teacher-forced val loss
        self.log = log
        self.swa_lr = cfg.wsc_swa_lr or cfg.learning_rate
        self.prev_state = None
        self.tracker = None
        self.watcher = None
        self.swa_model = None
        self.swa_sched = None
        self.swa_active = False
        self.swa_epochs = 0
        self.epochs_seen = 0                  # epoch-ends actually processed
        self.events = []                      # for summary.json

    def begin_task(self, model, t):
        self.swa_model = self.swa_sched = None
        self.swa_active = False
        self.swa_epochs = 0
        self.epochs_seen = 0
        self.tracker = MomentTracker(model)
        self.watcher = PlateauWatcher(self.cfg.wsc_patience, self.cfg.wsc_plateau_tol)
        self._t = t
        # theta(t-1) is the model as it stands BEFORE this task's training.
        self.prev_state = None if t == 0 else snapshot_params(model)
        if t == 0:
            self.log("[wsc] task 0: plain fine-tuning (no previous task to blend toward)")

    def on_optimizer_step(self, model):
        if self.tracker is not None:
            self.tracker.update(model)

    def on_epoch_end(self, model, optimizer, epoch, t):
        self.epochs_seen = max(self.epochs_seen, epoch)
        if t == 0:
            return
        was = model.training
        model.eval()
        try:
            val = self.val_loss_fn()
        finally:
            if was:
                model.train()
        fire = self.watcher.step(val)
        msg = (f"[wsc] task {t} epoch {epoch}: val_loss {val:.4f} "
               f"(best {self.watcher.best:.4f}, stagnant {self.watcher.stagnant})")
        if fire:
            n_t, n_w = pre_swa_trim(model, self.prev_state, self.tracker.scores(),
                                    self.cfg.wsc_retain_percent)
            self.swa_model, self.swa_sched = make_swa(
                model, optimizer, self.swa_lr, self.cfg.wsc_anneal_epochs)
            self.swa_active = True
            msg += (f" -> PLATEAU: trimmed {n_t} tensors ({n_w:,} weights blended "
                    f"50/50 toward theta(t-1), keeping top "
                    f"{self.cfg.wsc_retain_percent:g}%); SWA started at lr {self.swa_lr:g}")
            self.events.append({"task": t, "epoch": epoch, "val_loss": val,
                                "trimmed_tensors": n_t, "blended_weights": n_w})
        self.log(msg, flush=True)
        if self.swa_active:
            self.swa_model.update_parameters(model)
            self.swa_sched.step()
            self.swa_epochs += 1

    def finish_epochs(self, model, optimizer, t, n_epochs):
        """Process the final epoch's end.

        train_task is step-based: on_epoch_end fires when the data iterator
        exhausts, which never happens for the last epoch because the step loop
        finishes first. Upstream iterates epochs explicitly and validates after
        every one, including the last -- without this the final epoch is never
        averaged into SWA (observed: 1 SWA epoch where 2 were due).
        """
        n = int(n_epochs)
        while self.epochs_seen < n:
            self.on_epoch_end(model, optimizer, self.epochs_seen + 1, t)

    def end_task(self, model, t):
        if self.swa_active and self.swa_epochs > 0:
            load_swa_into(model, self.swa_model)
            self.log(f"[wsc] task {t}: loaded SWA average over {self.swa_epochs} epoch(s)")
        elif t > 0:
            self.log(f"[wsc] task {t}: SWA never triggered "
                     f"(stagnant reached {self.watcher.stagnant}/{self.cfg.wsc_patience}) "
                     f"-- this task was plain fine-tuning")
        self.swa_model = self.swa_sched = None
        self.swa_active = False
