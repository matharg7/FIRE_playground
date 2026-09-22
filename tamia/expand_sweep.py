#!/usr/bin/env python3
"""Expand a W&B grid sweep YAML into a flat trial list for offline execution.

tamIA's compute nodes have no internet, so `wandb agent` -- which asks the sweep
server for the next hyperparameter set -- cannot run there. This script does the
server's one job for a `method: grid` sweep, ahead of time, on a login node: it
takes the exact same `sweep_config/*.yaml` the online runs use and writes out
every point of the grid, in the same `--key=value` form that `${args}` expands to.

Usage
-----
    # by sweep name (resolved against ../sweep_config/)
    python tamia/expand_sweep.py tin_vgg16_class_inc_set_rigl

    # or by path, anywhere
    python tamia/expand_sweep.py sweep_config/tin_vgg16_class_inc_dense.yaml

    # inspect without writing
    python tamia/expand_sweep.py <sweep> --stdout

Output
------
`tamia/trials/<sweep>.tsv`, one trial per line, three tab-separated columns:

    <tid>   <label>   <args>

  tid    10 hex chars of sha1(args). Stable across re-expansions of an unchanged
         grid, different the moment the grid changes. The runner uses it as the
         claim/done marker name, which is what makes a sweep resumable.
  label  the swept (non-constant) parameters, for reading Slurm logs.
  args   the argument list, already shell-quoted. The runner consumes it with
         `eval set -- "$args"`, which is why `--wandb_project='DST Continual
         Learning'` survives the round trip.

Validation
----------
Every flag is checked against the options `bash_scripts/run_vision_st.sh` actually
accepts, parsed out of its own `case` block. A grid that sweeps something the
wrapper cannot take (`t_end_ratio`, say) fails here, in a second, on a login node
-- instead of failing 200 times inside a GPU allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import re
import shlex
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    sys.exit(
        "ERROR: PyYAML not importable.\n"
        "       Activate the venv first:  source tamia/env.sh"
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = REPO_ROOT / "sweep_config"
WRAPPER = REPO_ROOT / "bash_scripts" / "run_vision_st.sh"
TRIALS_DIR = REPO_ROOT / "tamia" / "trials"

# Injected per worker slot by run_sweep.slurm; a sweep must not also set it.
RUNNER_OWNED = {"gpu"}

# wandb's own command-template tokens, which are not constants to pass through.
COMMAND_TOKENS = {"${program}", "${args}", "${args_no_hyphens}", "${env}",
                  "${interpreter}", "${args_json}", "${args_json_file}"}


def wrapper_options(path: Path) -> set[str]:
    """Flags accepted by run_vision_st.sh, read from its `case` labels.

    Parsed rather than hardcoded so that adding an option to the wrapper does not
    silently leave this validator rejecting it.
    """
    if not path.is_file():
        sys.exit(f"ERROR: wrapper not found: {path}")

    opts: set[str] = set()
    for line in path.read_text().splitlines():
        m = re.match(r"^\s*(-[^)]*)\)", line)
        if not m:
            continue
        for label in m.group(1).split("|"):
            label = label.strip()
            if label.endswith("=*"):
                label = label[:-2]
            if label.startswith("--"):
                opts.add(label)
    if not opts:
        sys.exit(f"ERROR: parsed no options out of {path} -- has its case block changed?")
    return opts


def resolve_sweep(arg: str) -> Path:
    """Accept a sweep name, a bare filename, or any path to the YAML."""
    candidates = [Path(arg)]
    if not arg.endswith((".yaml", ".yml")):
        candidates.append(SWEEP_DIR / f"{arg}.yaml")
        candidates.append(SWEEP_DIR / f"{arg}.yml")
    else:
        candidates.append(SWEEP_DIR / Path(arg).name)

    for c in candidates:
        if c.is_file():
            return c.resolve()

    available = sorted(p.stem for p in SWEEP_DIR.glob("*.yaml"))
    sys.exit(
        f"ERROR: no sweep config for '{arg}'.\n"
        "       Available in sweep_config/:\n         "
        + "\n         ".join(available)
    )


def axes(params: dict, sweep_path: Path) -> list[tuple[str, list]]:
    """Turn the `parameters:` block into ordered (name, values) axes.

    Constants (`value:`) become one-element axes, so the cartesian product below
    handles constants and swept axes uniformly and the emitted argument order
    matches the YAML's.
    """
    out: list[tuple[str, list]] = []
    for name, spec in params.items():
        if name in RUNNER_OWNED:
            sys.exit(
                f"ERROR: {sweep_path.name} sets '{name}', which run_sweep.slurm assigns\n"
                f"       per worker slot. Remove it from the sweep's parameters."
            )
        if not isinstance(spec, dict):
            # `name: 3` shorthand -- wandb treats this as a constant.
            out.append((name, [spec]))
        elif "values" in spec:
            vals = spec["values"]
            if not isinstance(vals, list) or not vals:
                sys.exit(f"ERROR: {sweep_path.name}: '{name}.values' must be a non-empty list.")
            out.append((name, list(vals)))
        elif "value" in spec:
            out.append((name, [spec["value"]]))
        else:
            sys.exit(
                f"ERROR: {sweep_path.name}: parameter '{name}' has neither 'value' nor\n"
                f"       'values'. Only grid sweeps can be expanded offline -- a\n"
                f"       distribution needs the W&B server to sample it."
            )
    return out


def fmt(value) -> str:
    """Render a YAML scalar the way wandb's ${args} would."""
    if isinstance(value, bool):
        # The sweeps already quote these ("True"), but a bare YAML `true` would
        # otherwise reach config_st.str2bool as "true" -- which it accepts anyway.
        return "True" if value else "False"
    return str(value)


def trailing_constants(command: list, sweep_path: Path) -> list[str]:
    """The literal arguments after ${args} in the `command:` block.

    In these sweeps that is `--log-subdir <name>`, which is what puts each trial's
    stdout under logs/<subdir>/. Kept so offline runs land exactly where the
    online ones did.
    """
    if not command:
        return []
    consts: list[str] = []
    seen_args = False
    for entry in command:
        token = str(entry)
        if token in COMMAND_TOKENS:
            seen_args = token != "${program}" or seen_args
            continue
        if not seen_args:
            # `bash`, or the interpreter -- part of how wandb launches, not an arg.
            continue
        consts.append(token)
    return consts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Expand a W&B grid sweep YAML into a trial list for offline runs.")
    ap.add_argument("sweep", help="sweep name (e.g. tin_vgg16_class_inc_gmp) or path to the YAML")
    ap.add_argument("--stdout", action="store_true",
                    help="print the trials instead of writing tamia/trials/<sweep>.tsv")
    ap.add_argument("--out", type=Path, default=None, help="write somewhere other than the default")
    args = ap.parse_args()

    sweep_path = resolve_sweep(args.sweep)
    name = sweep_path.stem
    cfg = yaml.safe_load(sweep_path.read_text())

    method = cfg.get("method", "grid")
    if method != "grid":
        sys.exit(
            f"ERROR: {sweep_path.name} uses method '{method}'.\n"
            "       Only 'grid' can be expanded offline; bayes/random need the\n"
            "       W&B server, which tamIA's compute nodes cannot reach."
        )

    params = cfg.get("parameters") or {}
    if not params:
        sys.exit(f"ERROR: {sweep_path.name} has no 'parameters:' block.")

    allowed = wrapper_options(WRAPPER)
    consts = trailing_constants(cfg.get("command") or [], sweep_path)

    # Validate flag names once, up front, against the wrapper.
    unknown = []
    for pname, _ in axes(params, sweep_path):
        hyphen = "--" + pname.replace("_", "-")
        underscore = "--" + pname
        if hyphen not in allowed and underscore not in allowed:
            unknown.append(pname)
    for c in consts:
        if c.startswith("--") and c not in allowed:
            unknown.append(c.lstrip("-"))
    if unknown:
        sys.exit(
            f"ERROR: {sweep_path.name} sweeps parameters that "
            f"{WRAPPER.relative_to(REPO_ROOT)} does not accept:\n"
            "         " + ", ".join(sorted(set(unknown))) + "\n"
            "       Either drop them from the sweep or add the flags to the wrapper.\n"
            "       (Accepted: " + ", ".join(sorted(allowed)) + ")"
        )

    ax = axes(params, sweep_path)
    swept = {n for n, v in ax if len(v) > 1}

    lines = []
    seen: dict[str, str] = {}
    for combo in itertools.product(*(v for _, v in ax)):
        argv: list[str] = []
        label_bits: list[str] = []
        for (pname, _), value in zip(ax, combo):
            argv.append(f"--{pname}={fmt(value)}")
            if pname in swept:
                label_bits.append(f"{pname}={fmt(value)}")
        argv.extend(consts)

        quoted = " ".join(shlex.quote(a) for a in argv)
        tid = hashlib.sha1(quoted.encode()).hexdigest()[:10]
        if tid in seen:
            sys.exit(f"ERROR: trial id collision between:\n  {seen[tid]}\n  {quoted}")
        seen[tid] = quoted

        label = ",".join(label_bits) if label_bits else "single"
        lines.append(f"{tid}\t{label}\t{quoted}")

    header = (
        f"# sweep       : {name}\n"
        f"# source      : {sweep_path}\n"
        f"# project     : {cfg.get('project', '(unset)')}\n"
        f"# trials      : {len(lines)}\n"
        f"# swept axes  : {', '.join(sorted(swept)) or '(none)'}\n"
        f"# columns     : tid<TAB>label<TAB>args   (args is shell-quoted; "
        f"consume with `eval set -- \"$args\"`)\n"
    )

    if args.stdout:
        sys.stdout.write(header)
        for line in lines:
            print(line)
        return

    out = args.out or (TRIALS_DIR / f"{name}.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "\n".join(lines) + "\n")
    print(f"{len(lines)} trials -> {out}")
    if swept:
        print(f"swept axes: {', '.join(sorted(swept))}")


if __name__ == "__main__":
    main()
