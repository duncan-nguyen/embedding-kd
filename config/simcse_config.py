from .base_config import BaseConfig


class SimCSEConfig(BaseConfig):
    """SimCSE-only: the no-distillation control, teacher signal removed."""

    distill_method = "simcse"

    student_model_name = "bert-base-uncased"
    # Recorded so the run says which comparison it is the control for; no teacher
    # tokenizer and no teacher weights are loaded for this method.
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"

    # Second view of Gao et al. (2021): "dropout" runs the same sentence twice
    # under independent dropout masks (unsupervised SimCSE, the paper's setting);
    # "pair" uses the second sentence of the training row as the positive, which
    # turns the control into the supervised variant.
    simcse_view = "dropout"

    # tau = 0.05 is the unsupervised SimCSE setting.
    temperature = 0.05

    # "cls" is what the evaluation code reads. SimCSE trains an extra MLP head on
    # top of [CLS] and drops it at test time; the head is left out here so the
    # control deploys exactly the encoder every other row deploys.
    student_pooling = "cls"

    # Deliberately the shared schedule of the other methods, not SimCSE's own
    # (batch 64, lr 3e-5, 1 epoch): the control has to differ from the distilled
    # rows by the teacher term alone.
    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    min_lr = 2e-6


    save_dir = "checkpoints/simcse"
