class BaseConfig:
    
    task_type = "pair_cls"
    max_length = 256
    
    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    min_lr = 2e-6
    warmup_ratio = 0.06
    
    w_task = 0.5
    alpha_dtw = 0.5
    w_cls = 1.0
    temperature = 0.07
    
    student_model_name = "bert-base-uncased"
    teacher_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    teacher_dtype = "float32"
    # Pooling of the teacher sentence vector, for every method that has a teacher:
    # the cached methods (talas/geoode/rkd) apply it once at cache time, the online
    # ones (cdm/dskd/emo/stella) every step. "last_token" is the Qwen3-Embedding
    # convention; encoder teachers such as BGE-M3 read "cls". CLI: --teacher_pooling.
    pooling_method = "last_token"
    
    # Sub-word markers the token-level alignments strip before comparing token
    # strings: "##" for WordPiece students, "Ġ" for byte-level BPE teachers
    # (Qwen3), "▁" for SentencePiece teachers (BGE-M3). EMO reads
    # teacher_special_token as the teacher's BOS token string instead.
    # CLI: --student_special_token / --teacher_special_token.
    student_special_token = "##"
    teacher_special_token = "_"
    
    train_data_path = "data/train_set/merged_3_data_5k_each.csv"
    eval_data_path = None
    num_workers = 2
    
    distill_method = "cdm"
    
    save_dir = "checkpoints"
    weights_dir = None
    save_every = 1
    save_best = True
    
    # Per-depth diagnostics are only defined for methods with a cached teacher
    # embedding per example, so the base default is off.
    depth_log_every = 0

    # Where the pair-classification decision threshold is swept. "validation" keeps
    # the test score held out; "test" sweeps it on the test split itself, which turns
    # the pair accuracy/F1 into an upper bound rather than an estimate.
    pair_threshold_source = "validation"

    debug_align = False
    evaluate_test_each_epoch = False
    # Per-epoch evaluation cadence; 0 disables it (only the final test eval runs).
    eval_every = 1
    
    seed = 42
    
    def __repr__(self):
        attrs = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"{self.__class__.__name__}({', '.join(attrs)})"
    
    def to_dict(self):
        values = {}
        for cls in reversed(type(self).mro()):
            for key, value in vars(cls).items():
                if key.startswith("_") or callable(value):
                    continue
                values[key] = value
        values.update(
            {
                key: value
                for key, value in self.__dict__.items()
                if not key.startswith("_")
            }
        )
        return values
