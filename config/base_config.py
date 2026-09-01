class BaseConfig:
    """Defaults shared by every method; a subclass overrides only what it changes.

    Keyword arguments to the constructor override any attribute the class already
    defines. An unknown name is ignored rather than raising, so a caller may pass a
    superset of settings and each config takes the ones it understands.
    """

    task_type = "pair_cls"
    max_length = 256

    batch_size = 64
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

    # Every method in the paper is distilled on this one corpus: 100k benchmark
    # sentences plus 25k MS MARCO queries and 25k MS MARCO passages, all raw
    # text. Built by scripts/build_train_corpus.py.
    train_data_path = "data/train_set/train_150k.csv"
    # Shared directory for the cached teacher embeddings of talas/geoode/rkd. When
    # set it wins over cache_path, and the filename is derived from what a cache's
    # reusability actually depends on (teacher, pooling, normalisation, max_length
    # and the corpus *contents*), so one directory holds every cache the project
    # builds and a run either finds exactly its own or misses. Point it somewhere
    # that outlives a single run -- re-encoding 100k sentences with a 4B-parameter
    # teacher is the most expensive thing in the pipeline and never changes between
    # runs of the same pair. CLI: --cache_dir.
    cache_dir = None
    # Batch size of the one-off teacher pass that builds that cache. It is not the
    # training batch size: nothing in that pass is kept for backward, and the
    # batches are formed over length-sorted rows, so it fits far more rows at once.
    # 0 falls back to batch_size. CLI: --cache_batch_size.
    cache_batch_size = 128
    eval_data_path = None
    num_workers = 2
    # Encode both dropout views of a batch in one forward over the doubled batch
    # rather than in two forwards over the batch. Same pair of views, same FLOPs,
    # half the kernel launches -- but one RNG draw instead of two, so a seeded run
    # follows a different (equally valid) trajectory. --no-fused_views restores the
    # two-pass order for reproducing numbers collected before it.
    fused_views = True

    distill_method = "cdm"

    save_dir = "checkpoints"
    weights_dir = None
    save_every = 1
    save_best = True

    # Where the pair-classification decision threshold is swept. "validation" keeps
    # the test score held out; "test" sweeps it on the test split itself, which turns
    # the pair accuracy/F1 into an upper bound rather than an estimate.
    pair_threshold_source = "validation"

    # Score ArguAna/FiQA/SCIDOCS (nDCG@10) as part of the final test evaluation.
    # Needs data/test_set/retrieval, written by
    # scripts/download_retrieval_benchmarks.py. CLI: --no_eval_retrieval.
    eval_retrieval = True

    debug_align = False
    evaluate_test_each_epoch = False
    # Per-epoch evaluation cadence; 0 disables it (only the final test eval runs).
    eval_every = 1

    seed = 42

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

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
