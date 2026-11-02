"""defectnet-train: TACE's tace-train with the periodic QEq electrostatics of DefectNet registered first.

Use it instead of tace-train to fine-tune DefectNet-v1.0.pt; the arguments are the same (-cn finetune.yaml and
Hydra overrides). Without defectnet.qeq the QEq head would treat every training cell as an open cluster."""


def main():
    from . import qeq  # noqa: F401
    from tace.scripts.train import main as train
    train()


if __name__ == "__main__":
    main()
