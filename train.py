"""
train.py
Multi-task training loop for the DefectNet force-field model.

Loss =  w_e · MSE(E/N)  +  w_f · MSE(F)  +  w_s · MSE(σ)

Usage
-----
  python train.py --csv /path/to/merged_data.csv \\
                  --epochs 200 --batch 4 --lr 1e-3

  python train.py --csv merged_data.csv --device cuda --epochs 300 \\
                  --weight_force 50 --weight_stress 0.1

  # resume from checkpoint
  python train.py --csv merged_data.csv --restart trained_model/last.pt
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import create_dataloaders, KBAR_TO_EV_PER_A3
from model import DefectNetForceField


# -----------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train DefectNet Force Field")

    # data
    p.add_argument("--csv", required=True, help="Path to merged_data.csv")
    p.add_argument("--cache_dir", default=None,
                   help="Directory for caching processed graphs")
    p.add_argument("--stress_unit", default="kbar",
                   choices=["kbar", "ev_per_a3"],
                   help="Unit of stress in CSV (default: kbar)")

    # splits
    p.add_argument("--fraction",    type=float, default=1.0,
                   help="Fraction of data to use (0<f<=1, e.g. 0.01 = 1%%)")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio",   type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=42)

    # model
    p.add_argument("--atom_fea_len",  type=int,   default=64)
    p.add_argument("--num_conv",      type=int,   default=4)
    p.add_argument("--num_gaussians", type=int,   default=80)
    p.add_argument("--cutoff",        type=float, default=5.0)
    p.add_argument("--max_neighbors", type=int,   default=12)

    # training
    p.add_argument("--epochs", type=int,   default=200)
    p.add_argument("--batch",  type=int,   default=4)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=0)

    # loss weights
    p.add_argument("--weight_energy", type=float, default=1.0)
    p.add_argument("--weight_force",  type=float, default=50.0)
    p.add_argument("--weight_stress", type=float, default=0.1)

    # output
    p.add_argument("--outdir",  default="trained_model")
    p.add_argument("--restart", default=None,
                   help="Checkpoint to resume from")

    return p.parse_args()


# -----------------------------------------------------------------------
#  Atomic save  (write to .tmp, then atomic rename)
# -----------------------------------------------------------------------
def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# -----------------------------------------------------------------------
#  Move batch to device
# -----------------------------------------------------------------------
def to_device(data, device):
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in data.items()
    }


# -----------------------------------------------------------------------
#  Compute multi-task loss
# -----------------------------------------------------------------------
def compute_loss(pred, data, w_e, w_f, w_s):
    """
    Returns (total_loss, loss_energy, loss_force, loss_stress)
    all as Python floats except total_loss which keeps the graph.
    """
    mse = nn.functional.mse_loss
    num_atoms = data["num_atoms"].float()

    # per-atom energy MSE
    e_pred = pred["energy"] / num_atoms
    e_true = data["energy"] / num_atoms
    loss_e = mse(e_pred, e_true)

    # force MSE  (per-component)
    loss_f = mse(pred["forces"], data["forces"])

    # stress MSE (per-component)
    loss_s = mse(pred["stress"], data["stress"])

    total = w_e * loss_e + w_f * loss_f + w_s * loss_s
    return total, loss_e.item(), loss_f.item(), loss_s.item()


# -----------------------------------------------------------------------
#  One epoch
# -----------------------------------------------------------------------
def run_epoch(model, loader, optimizer, device, w_e, w_f, w_s, training=True):
    if training:
        model.train()
    else:
        model.eval()

    sum_loss, sum_e, sum_f, sum_s = 0.0, 0.0, 0.0, 0.0
    n_batch = 0

    for data in loader:
        data = to_device(data, device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # always need grads for forces/stress even in eval
            pred = model(data, compute_forces=True, compute_stress=True)

        loss, le, lf, ls = compute_loss(pred, data, w_e, w_f, w_s)

        if training:
            loss.backward()
            # gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        sum_loss += loss.item()
        sum_e += le
        sum_f += lf
        sum_s += ls
        n_batch += 1

    n = max(n_batch, 1)
    return sum_loss / n, sum_e / n, sum_f / n, sum_s / n


# -----------------------------------------------------------------------
#  Get the base ForceFieldDataset from a DataLoader (traverses Subsets)
# -----------------------------------------------------------------------
def _get_base_df(loader):
    ds = loader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds.df


# -----------------------------------------------------------------------
#  Save per-structure predictions vs DFT for a given split
# -----------------------------------------------------------------------
def save_predictions(model, loader, device, outdir, split_name, orig_df):
    """
    Iterate over a DataLoader, collect predicted vs DFT energy / forces /
    stress, and save to ``{outdir}/{split_name}_predictions.csv``.

    Each row includes Structure, Charge, LevelOfTheory from the original CSV.
    """
    model.eval()

    rows = []
    force_preds, force_trues = [], []
    stress_preds, stress_trues = [], []

    for data in loader:
        data = to_device(data, device)

        with torch.set_grad_enabled(True):
            pred = model(data, compute_forces=True, compute_stress=True)

        e_pred = pred["energy"].detach().cpu().numpy()
        e_true = data["energy"].cpu().numpy()
        natoms = data["num_atoms"].cpu().numpy()
        idxs   = data["idx"].cpu().numpy()

        f_pred = pred["forces"].detach().cpu().numpy()
        f_true = data["forces"].cpu().numpy()
        s_pred = pred["stress"].detach().cpu().numpy()
        s_true = data["stress"].cpu().numpy()

        # accumulate for aggregate metrics
        force_preds.append(f_pred)
        force_trues.append(f_true)
        stress_preds.append(s_pred)
        stress_trues.append(s_true)

        # split forces/stress per structure
        atom_cursor = 0
        for i in range(len(natoms)):
            n = natoms[i]
            fp = f_pred[atom_cursor:atom_cursor + n].tolist()
            ft = f_true[atom_cursor:atom_cursor + n].tolist()
            sp = s_pred[i].tolist()
            st = s_true[i].tolist()

            csv_i = int(idxs[i])
            orig_row = orig_df.iloc[csv_i]

            rows.append({
                "csv_idx":              csv_i,
                "Structure":            orig_row["Structure"],
                "Charge":               orig_row["Charge"],
                "LevelOfTheory":        orig_row["LevelOfTheory"],
                "num_atoms":            int(n),
                "energy_dft":           float(e_true[i]),
                "energy_pred":          float(e_pred[i]),
                "energy_err_per_atom":  float((e_pred[i] - e_true[i]) / n),
                "forces_dft":           str(ft),
                "forces_pred":          str(fp),
                "stress_dft_eV_A3":     str(st),
                "stress_pred_eV_A3":    str(sp),
            })
            atom_cursor += n

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, f"{split_name}_predictions.csv")
    df.to_csv(csv_path, index=False)

    # aggregate metrics
    all_fp = np.concatenate(force_preds)
    all_ft = np.concatenate(force_trues)
    all_sp = np.concatenate(stress_preds)
    all_st = np.concatenate(stress_trues)

    e_arr_pred = df["energy_pred"].values
    e_arr_true = df["energy_dft"].values
    n_arr      = df["num_atoms"].values

    e_mae  = np.mean(np.abs((e_arr_pred - e_arr_true) / n_arr))
    e_rmse = np.sqrt(np.mean(((e_arr_pred - e_arr_true) / n_arr) ** 2))
    f_mae  = np.mean(np.abs(all_fp - all_ft))
    f_rmse = np.sqrt(np.mean((all_fp - all_ft) ** 2))
    s_mae  = np.mean(np.abs(all_sp - all_st))
    s_rmse = np.sqrt(np.mean((all_sp - all_st) ** 2))

    print(f"  {split_name:5s}  ({len(df)} structures)  →  {csv_path}")
    print(f"         Energy MAE={e_mae:.5f}  RMSE={e_rmse:.5f} eV/atom")
    print(f"         Force  MAE={f_mae:.6f} RMSE={f_rmse:.6f} eV/A")
    print(f"         Stress MAE={s_mae:.6f} RMSE={s_rmse:.6f} eV/A^3")


# -----------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # ---- data ----
    print("Loading data …")
    train_dl, val_dl, test_dl = create_dataloaders(
        csv_path=args.csv,
        batch_size=args.batch,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        cache_dir=args.cache_dir,
        stress_unit=args.stress_unit,
        seed=args.seed,
        num_workers=args.num_workers,
        fraction=args.fraction,
    )
    print(f"  train={len(train_dl.dataset)}  "
          f"val={len(val_dl.dataset)}  "
          f"test={len(test_dl.dataset)}")

    # ---- model ----
    model = DefectNetForceField(
        atom_fea_len=args.atom_fea_len,
        num_conv=args.num_conv,
        num_gaussians=args.num_gaussians,
        cutoff=args.cutoff,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6,
    )

    start_epoch = 1
    best_val = float("inf")

    # ---- optional restart ----
    if args.restart is not None:
        print(f"Loading checkpoint: {args.restart}")
        ckpt = torch.load(args.restart, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"]) + 1
            if "best_val" in ckpt:
                best_val = ckpt["best_val"]
        else:
            model.load_state_dict(ckpt)
        print(f"  Resuming at epoch {start_epoch}")

    # ---- save config ----
    cfg = vars(args)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- training loop ----
    w_e, w_f, w_s = args.weight_energy, args.weight_force, args.weight_stress

    print(f"\nTraining for {args.epochs} epochs  "
          f"(w_e={w_e}, w_f={w_f}, w_s={w_s})\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_e, tr_f, tr_s = run_epoch(
            model, train_dl, optimizer, device, w_e, w_f, w_s, training=True,
        )

        with torch.no_grad():
            vl_loss, vl_e, vl_f, vl_s = run_epoch(
                model, val_dl, None, device, w_e, w_f, w_s, training=False,
            )

        scheduler.step(vl_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(
            f"[{time.strftime('%H:%M')}] Epoch {epoch:03d}  "
            f"({dt:.0f}s)  lr={lr_now:.1e}\n"
            f"  Train  loss={tr_loss:.5f}  "
            f"E={tr_e:.5f}  F={tr_f:.6f}  S={tr_s:.6f}\n"
            f"  Val    loss={vl_loss:.5f}  "
            f"E={vl_e:.5f}  F={vl_f:.6f}  S={vl_s:.6f}"
        )

        # ---- always save last ----
        _atomic_save(
            {
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch":           epoch,
                "best_val":        best_val,
                "config":          cfg,
            },
            os.path.join(args.outdir, "last.pt"),
        )

        # ---- save best ----
        if vl_loss < best_val:
            best_val = vl_loss
            _atomic_save(
                {
                    "model_state": model.state_dict(),
                    "epoch":       epoch,
                    "best_val":    best_val,
                    "config":      cfg,
                },
                os.path.join(args.outdir, "best.pt"),
            )
            print(f"  ** new best (val loss = {best_val:.5f})")

    # ---- final test ----
    print("\nEvaluating on test set …")
    best_ckpt = torch.load(
        os.path.join(args.outdir, "best.pt"), map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_ckpt["model_state"])

    with torch.no_grad():
        ts_loss, ts_e, ts_f, ts_s = run_epoch(
            model, test_dl, None, device, w_e, w_f, w_s, training=False,
        )
    print(
        f"Test  loss={ts_loss:.5f}  "
        f"E={ts_e:.5f}  F={ts_f:.6f}  S={ts_s:.6f}\n"
        f"  E RMSE = {ts_e**0.5:.5f} eV/atom\n"
        f"  F RMSE = {ts_f**0.5:.6f} eV/A\n"
        f"  S RMSE = {ts_s**0.5:.6f} eV/A^3"
    )

    # ---- save train / val / test prediction CSVs ----
    print("\nSaving prediction CSVs …")
    orig_df = _get_base_df(train_dl)
    for split_name, loader in [("train", train_dl),
                                ("val",   val_dl),
                                ("test",  test_dl)]:
        save_predictions(model, loader, device, args.outdir, split_name,
                         orig_df)

    print(f"\nDone.  Best checkpoint: {args.outdir}/best.pt")


if __name__ == "__main__":
    main()
