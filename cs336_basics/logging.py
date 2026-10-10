import csv, logging
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curves(run_dir):
    with open(run_dir / "metrics.csv", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return

    steps = [int(row["step"]) for row in rows]
    minutes = [float(row["elapsed_seconds"]) / 60 for row in rows]
    train_losses = [float(row["train_loss"]) for row in rows]
    val_losses = [float(row["val_loss"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, x, xlabel in [
        (axes[0], steps, "Gradient step"),
        (axes[1], minutes, "Elapsed time (min)"),
    ]:
        ax.plot(x, train_losses, label="Train")
        ax.plot(x, val_losses, label="Validation")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    save_path = run_dir / "loss_curves.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    logging.info("Loss curves saved to %s", save_path)