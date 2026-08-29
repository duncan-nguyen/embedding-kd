########################################################################################
METHOD 1/3: GEOODE
Log: /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/geoode_train.log
########################################################################################
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
======================================================================
Configuration for GEOODE method:
======================================================================
  task_type                 : pair_cls
  max_length                : 256
  batch_size                : 64
  epochs                    : 5
  learning_rate             : 5e-05
  min_lr                    : 2e-06
  warmup_ratio              : 0.06
  w_task                    : 0.5
  alpha_dtw                 : 0.5
  w_cls                     : 1.0
  temperature               : 0.07
  student_model_name        : nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base
  teacher_model_name        : Qwen/Qwen3-Embedding-0.6B
  teacher_dtype             : bfloat16
  pooling_method            : last_token
  student_special_token     : ##
  teacher_special_token     : G
  train_data_path           : /content/embedding-kd/data/train_set/train_100k.csv
  cache_dir                 : /content/embedding-kd/runs/teacher_cache
  eval_data_path            : None
  num_workers               : 2
  distill_method            : geoode
  save_dir                  : /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/geoode
  weights_dir               : None
  save_every                : 5
  save_best                 : True
  pair_threshold_source     : test
  eval_retrieval            : False
  debug_align               : False
  evaluate_test_each_epoch  : True
  eval_every                : 0
  seed                      : 42
  lambda_end                : 1.0
  lambda_ctr                : 0.5
  contrastive_temperature   : 0.05
  endpoint_loss             : cosine
  student_pooling           : cls
  include_embedding_layer   : False
  contrastive_view          : dropout
  eps_norm                  : 1e-12
  projection_type           : pca
  projection_seed           : 0
  learned_projector_lr_scale : 1.0
  pca_center_fit            : True
  pca_subtract_mean         : False
  gauge_align               : True
  gauge_align_samples       : 16384
  gauge_rotation            : procrustes
  gauge_random_seed         : 0
  gauge_refit_every         : 0
  cache_teacher             : True
  cache_path                : cache/teacher_train.pt
  normalize_cache           : True
  cache_dtype               : float32
  use_wandb                 : False
======================================================================
Per-epoch evaluation runs on the TEST split and the pair threshold is swept on it. No validation pass will run, so no number this run reports is held out.
Done setup_seed with seed=42
[WARN] Only 1 GPU available -> both on cuda:0
Done setup_devices
Loading tokenizers...
Loading student model: nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base
Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]                                                                                                        
[transformers] [1mBertModel LOAD REPORT[0m from: nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.predictions.decoder.weight             | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 
Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Loading teacher model: Qwen/Qwen3-Embedding-0.6B
Teacher pooling: last_token (dim 1024)
Student training dtype: torch.float32
Models loaded successfully!
Done setup_models
Loading training data from: /content/embedding-kd/data/train_set/train_100k.csv
Teacher cache (shared directory): /content/embedding-kd/runs/teacher_cache/qwen-qwen3-embedding-0-6b__train_100k__last_token__383e4d6c7e18.pt
Cache not found. Pre-computing teacher embeddings...
Pre-computing teacher embeddings...
Caching teacher CLS embeddings:   0%|          | 0/1600 [00:00<?, ?it/s]                                                                                        
Saved cached teacher embeddings to: /content/embedding-kd/runs/teacher_cache/qwen-qwen3-embedding-0-6b__train_100k__last_token__383e4d6c7e18.pt
Done caching teacher embeddings: torch.Size([102361, 1024])
Cached 102361 teacher embeddings to /content/embedding-kd/runs/teacher_cache/qwen-qwen3-embedding-0-6b__train_100k__last_token__383e4d6c7e18.pt
Fitted pca teacher projection 1024 -> 384 (retains 91.8% of cached embedding energy; a random subspace retains ~37.5%)
Fitted procrustes gauge rotation on 16384 sentences: mean student-target cosine +0.031 -> +0.620
Cross-covariance participation ratio 1.37 of 384 (top singular share 0.853)
Teacher projection saved: /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/geoode/teacher_projection.pt
Teacher model freed from GPU memory
Training samples: 102361
Training batches: 1600
Done setup_data
Checkpoints will be saved to: /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/geoode
Done setup_training
GeoODE-KD criterion initialized: lambda_end=1.0, lambda_ctr=0.5, endpoint_loss=cosine
============================================================
Starting training...
============================================================
Method: geoode
Student: nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base
Teacher: Qwen/Qwen3-Embedding-0.6B
Epochs: 5
Batch size: 64
Learning rate: 5e-05
============================================================
  self.scheduler.step()
Epoch 1/5:  81%|████████  | 1294/1600 [01:01<00:14, 21.57it/s, avg_loss=0.3858, ms/step=44.9, ms/step(ma)=43.9, it/s=22.80, gpu0=368/4162MB, loss_end=0.2534, lo
[Epoch 1] Avg step time = 44.82 ms (22.31 it/s)
Done train_epoch 1
[Epoch 1] mean over 102361 examples: loss_total=0.3636  cos_final=0.7063  cos_first=0.2622  loss_ctr=0.1397  loss_end=0.2937
============================================================
Evaluation after Epoch 1
============================================================
============================================================
Epoch 2/5:  41%|████      | 655/1600 [00:48<00:43, 21.81it/s, avg_loss=0.2389, ms/step=44.7, ms/step(ma)=43.4, it/s=23.03, gpu0=366/4234MB, loss_end=0.2228, los
[Epoch 2] Avg step time = 44.52 ms (22.46 it/s)
Done train_epoch 2
[Epoch 2] mean over 102361 examples: loss_total=0.2344  cos_final=0.7739  cos_first=0.2619  loss_ctr=0.0168  loss_end=0.2261
============================================================
Evaluation after Epoch 2
============================================================
============================================================
Epoch 3/5:  82%|████████▏ | 1319/1600 [01:07<00:12, 22.00it/s, avg_loss=0.2137, ms/step=44.5, ms/step(ma)=44.0, it/s=22.71, gpu0=366/4234MB, loss_end=0.2059, lo
[Epoch 3] Avg step time = 44.45 ms (22.50 it/s)
Done train_epoch 3
[Epoch 3] mean over 102361 examples: loss_total=0.2133  cos_final=0.7937  cos_first=0.2623  loss_ctr=0.0141  loss_end=0.2063
============================================================
Evaluation after Epoch 3
============================================================
============================================================
Epoch 4/5:  41%|████      | 654/1600 [00:55<00:43, 21.79it/s, avg_loss=0.2032, ms/step=44.6, ms/step(ma)=48.5, it/s=20.62, gpu0=368/4234MB, loss_end=0.1896, los
[Epoch 4] Avg step time = 44.68 ms (22.38 it/s)
Done train_epoch 4
[Epoch 4] mean over 102361 examples: loss_total=0.2027  cos_final=0.8036  cos_first=0.2626  loss_ctr=0.0125  loss_end=0.1964
============================================================
Evaluation after Epoch 4
============================================================
============================================================
Epoch 5/5:  82%|████████▏ | 1304/1600 [01:13<00:13, 21.73it/s, avg_loss=0.1985, ms/step=44.7, ms/step(ma)=44.7, it/s=22.39, gpu0=366/4234MB, loss_end=0.1955, lo
/content/embedding-kd/src/evaluation/evaluation_automodel.py:295: RuntimeWarning: Published test split banking77_test has normalized-text overlap: train=7, validation=0.
  _validate_classification_pair(train_path, dev_path)
[Epoch 5] Avg step time = 44.70 ms (22.37 it/s)
Done train_epoch 5
[Epoch 5] mean over 102361 examples: loss_total=0.1985  cos_final=0.8077  cos_first=0.2626  loss_ctr=0.0123  loss_end=0.1923
============================================================
Evaluation after Epoch 5
============================================================
============================================================
Checkpoint saved: /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/geoode/checkpoint_epoch_5.pt
Done save_checkpoint for epoch 5
Best model saved: /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/geoode/best_model.pt
============================================================
Training completed!
============================================================
Done train()
Checkpoint for epoch 5 already saved; skipping duplicate.
 eval classifier
data/test_set/banking77_test.csv
{'accuracy': 0.9154746423927178, 'f1': 0.9151923863613948}
data/test_set/emotion_test.csv
/content/embedding-kd/src/evaluation/evaluation_automodel.py:295: RuntimeWarning: Published test split tweet_test has normalized-text overlap: train=14, validation=0.
  _validate_classification_pair(train_path, dev_path)
{'accuracy': 0.74269889224572, 'f1': 0.6556166472527332}
data/test_set/tweet_test.csv
tweet_train:   0%|          | 0/93 [00:00<?, ?it/s]                                                                                                             
{'accuracy': 0.7400932400932401, 'f1': 0.7440452875766902}
 eval_pair_task
data/test_set/mrpc_test.csv
{'best_threshold': 0.9195979899497487, 'accuracy': 0.7286956521739131, 'f1': 0.626910506584639, 'precision': 0.7265621134913175, 'recall': 0.6243366929827472, 'average_precision': np.float64(0.8520791294319454)}
data/test_set/scitail_test.csv
{'best_threshold': 0.9246231155778895, 'accuracy': 0.8019755409219191, 'f1': 0.7917546041388681, 'precision': 0.7937771495003603, 'recall': 0.7900655611546459, 'average_precision': np.float64(0.8249876512657578)}
data/test_set/wic_test.csv
{'best_threshold': 0.8190954773869347, 'accuracy': 0.6371428571428571, 'f1': 0.6350507295906533, 'precision': 0.6403614306840113, 'recall': 0.6371428571428571, 'average_precision': np.float64(0.6706124964533767)}
 eval_sts_task
data/test_set/sick_test.csv
Spearman: 0.7717
data/test_set/sts12_test.csv
Spearman: 0.7235
data/test_set/stsb_test.csv
Spearman: 0.7763
=================================================================================================================
FINAL TEST  (pair thresholds calibrated on the test split)
=================================================================================================================
Family         | Benchmark | Primary metric    | Score | Details                                                 
---------------+-----------+-------------------+-------+---------------------------------------------------------
classification | banking77 | f1                | 91.52 | Acc=91.55 F1=91.52                                      
classification | emotion   | f1                | 65.56 | Acc=74.27 F1=65.56                                      
classification | tweet     | f1                | 74.40 | Acc=74.01 F1=74.40                                      
pair           | mrpc      | average_precision | 85.21 | Acc=72.87 F1=62.69 P=72.66 R=62.43 AP=85.21             
pair           | scitail   | average_precision | 82.50 | Acc=80.20 F1=79.18 P=79.38 R=79.01 AP=82.50             
pair           | wic       | average_precision | 67.06 | Acc=63.71 F1=63.51 P=64.04 R=63.71 AP=67.06             
sts            | sick      | spearman          | 77.17 | Spearman=77.17                                          
sts            | sts12     | spearman          | 72.35 | Spearman=72.35                                          
sts            | stsb      | spearman          | 77.63 | Spearman=77.63                                          
---------------+-----------+-------------------+-------+---------------------------------------------------------
summary        | AVG (IOD) | mean              | 70.08 | emotion stsb wic                                        
summary        | AVG (OOD) | mean              | 80.53 | banking77 mrpc scitail sick sts12 tweet                 
summary        | AVG (ALL) | mean              | 77.05 | banking77 emotion mrpc scitail sick sts12 stsb tweet wic
=================================================================================================================
[COMPLETE] geoode in 10.7 minutes

########################################################################################
METHOD 2/3: PCA_MSE
Log: /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/pca_mse_train.log
########################################################################################
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
======================================================================
Configuration for GEOODE method:
======================================================================
  task_type                 : pair_cls
  max_length                : 256
  batch_size                : 64
  epochs                    : 5
  learning_rate             : 5e-05
  min_lr                    : 2e-06
  warmup_ratio              : 0.06
  w_task                    : 0.5
  alpha_dtw                 : 0.5
  w_cls                     : 1.0
  temperature               : 0.07
  student_model_name        : nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base
  teacher_model_name        : Qwen/Qwen3-Embedding-0.6B
  teacher_dtype             : bfloat16
  pooling_method            : last_token
  student_special_token     : ##
  teacher_special_token     : G
  train_data_path           : /content/embedding-kd/data/train_set/train_100k.csv
  cache_dir                 : /content/embedding-kd/runs/teacher_cache
  eval_data_path            : None
  num_workers               : 2
  distill_method            : geoode
  save_dir                  : /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/pca_mse
  weights_dir               : None
  save_every                : 5
  save_best                 : True
  pair_threshold_source     : test
  eval_retrieval            : False
  debug_align               : False
  evaluate_test_each_epoch  : True
  eval_every                : 0
  seed                      : 42
  lambda_end                : 1.0
  lambda_ctr                : 0.0
  contrastive_temperature   : 0.05
  endpoint_loss             : mse
  student_pooling           : cls
  include_embedding_layer   : False
  contrastive_view          : dropout
  eps_norm                  : 1e-12
  projection_type           : pca
  projection_seed           : 0
  learned_projector_lr_scale : 1.0
  pca_center_fit            : True
  pca_subtract_mean         : False
  gauge_align               : False
  gauge_align_samples       : 16384
  gauge_rotation            : procrustes
  gauge_random_seed         : 0
  gauge_refit_every         : 0
  cache_teacher             : True
  cache_path                : cache/teacher_train.pt
  normalize_cache           : True
  cache_dtype               : float32
  use_wandb                 : False
======================================================================
Per-epoch evaluation runs on the TEST split and the pair threshold is swept on it. No validation pass will run, so no number this run reports is held out.
Done setup_seed with seed=42
[WARN] Only 1 GPU available -> both on cuda:0
Done setup_devices
Loading tokenizers...
Loading student model: nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base
Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]                                                                                                        
[transformers] [1mBertModel LOAD REPORT[0m from: nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.decoder.weight             | UNEXPECTED |  | 
Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Loading teacher model: Qwen/Qwen3-Embedding-0.6B
Teacher pooling: last_token (dim 1024)
Student training dtype: torch.float32
Models loaded successfully!
Done setup_models
Loading training data from: /content/embedding-kd/data/train_set/train_100k.csv
Teacher cache (shared directory): /content/embedding-kd/runs/teacher_cache/qwen-qwen3-embedding-0-6b__train_100k__last_token__383e4d6c7e18.pt
Loading cached teacher embeddings from: /content/embedding-kd/runs/teacher_cache/qwen-qwen3-embedding-0-6b__train_100k__last_token__383e4d6c7e18.pt
Loading cached embeddings from: /content/embedding-kd/runs/teacher_cache/qwen-qwen3-embedding-0-6b__train_100k__last_token__383e4d6c7e18.pt
Loaded embeddings: (102361, 1024)
Loaded 102361 cached embeddings (teacher not run for this training)
Fitted pca teacher projection 1024 -> 384 (retains 91.8% of cached embedding energy; a random subspace retains ~37.5%)
Teacher projection saved: /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/pca_mse/teacher_projection.pt
Teacher model freed from GPU memory
Training samples: 102361
Training batches: 1600
Done setup_data
Checkpoints will be saved to: /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/pca_mse
Done setup_training
GeoODE-KD criterion initialized: lambda_end=1.0, lambda_ctr=0.0, endpoint_loss=mse
============================================================
Starting training...
============================================================
Method: geoode
Student: nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base
Teacher: Qwen/Qwen3-Embedding-0.6B
Epochs: 5
Batch size: 64
Learning rate: 5e-05
============================================================
Epoch 1/5:   0%|          | 0/1600 [00:29<?, ?it/s, avg_loss=0.1234, ms/step=26.4, ms/step(ma)=25.8, it/s=38.76, gpu0=374/2512MB, loss_end=0.0842, loss_ctr=0.00
[Epoch 1] Avg step time = 26.31 ms (38.01 it/s)
Done train_epoch 1
[Epoch 1] mean over 102361 examples: loss_total=0.1063  cos_final=0.3873  cos_first=0.0033  loss_ctr=0.0000  loss_end=0.1063
============================================================
Evaluation after Epoch 1
============================================================
============================================================
Epoch 2/5:   0%|          | 0/1600 [00:15<?, ?it/s, avg_loss=0.0541, ms/step=26.3, ms/step(ma)=26.7, it/s=37.50, gpu0=372/2512MB, loss_end=0.0405, loss_ctr=0.00
[Epoch 2] Avg step time = 26.27 ms (38.07 it/s)
Done train_epoch 2
[Epoch 2] mean over 102361 examples: loss_total=0.0369  cos_final=0.2322  cos_first=0.0056  loss_ctr=0.0000  loss_end=0.0369
============================================================
Evaluation after Epoch 2
============================================================
============================================================
Epoch 3/5:  68%|██████▊   | 1094/1600 [00:31<00:13, 36.44it/s, avg_loss=0.0149, ms/step=26.2, ms/step(ma)=25.7, it/s=38.89, gpu0=370/2530MB, loss_end=0.0115, lo
[Epoch 3] Avg step time = 26.18 ms (38.19 it/s)
Done train_epoch 3
[Epoch 3] mean over 102361 examples: loss_total=0.0137  cos_final=0.2897  cos_first=0.0081  loss_ctr=0.0000  loss_end=0.0137
============================================================
Evaluation after Epoch 3
============================================================
============================================================
Epoch 4/5:   0%|          | 0/1600 [00:18<?, ?it/s, avg_loss=0.0088, ms/step=26.3, ms/step(ma)=27.0, it/s=36.99, gpu0=371/2530MB, loss_end=0.0079, loss_ctr=0.00
[Epoch 4] Avg step time = 26.31 ms (38.01 it/s)
Done train_epoch 4
[Epoch 4] mean over 102361 examples: loss_total=0.0078  cos_final=0.3581  cos_first=0.0097  loss_ctr=0.0000  loss_end=0.0078
============================================================
Evaluation after Epoch 4
============================================================
============================================================
Epoch 5/5:  68%|██████▊   | 1082/1600 [00:33<00:14, 36.06it/s, avg_loss=0.0061, ms/step=26.3, ms/step(ma)=26.3, it/s=38.04, gpu0=373/2530MB, loss_end=0.0059, lo
/content/embedding-kd/src/evaluation/evaluation_automodel.py:295: RuntimeWarning: Published test split banking77_test has normalized-text overlap: train=7, validation=0.
  _validate_classification_pair(train_path, dev_path)
[Epoch 5] Avg step time = 26.32 ms (37.99 it/s)
Done train_epoch 5
[Epoch 5] mean over 102361 examples: loss_total=0.0060  cos_final=0.3967  cos_first=0.0105  loss_ctr=0.0000  loss_end=0.0060
============================================================
Evaluation after Epoch 5
============================================================
============================================================
Checkpoint saved: /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/pca_mse/checkpoint_epoch_5.pt
Done save_checkpoint for epoch 5
Best model saved: /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/pca_mse/best_model.pt
============================================================
Training completed!
============================================================
Done train()
Checkpoint for epoch 5 already saved; skipping duplicate.
 eval classifier
data/test_set/banking77_test.csv
banking77_train:   0%|          | 0/36 [00:00<?, ?it/s]                                                                                                         
{'accuracy': 0.80851755526658, 'f1': 0.7944685320628708}
data/test_set/emotion_test.csv
/content/embedding-kd/src/evaluation/evaluation_automodel.py:295: RuntimeWarning: Published test split tweet_test has normalized-text overlap: train=14, validation=0.
  _validate_classification_pair(train_path, dev_path)
{'accuracy': 0.6883182275931521, 'f1': 0.5427383281015389}
data/test_set/tweet_test.csv
{'accuracy': 0.7293123543123543, 'f1': 0.7331537727758056}
 eval_pair_task
data/test_set/mrpc_test.csv
{'best_threshold': 0.9849246231155779, 'accuracy': 0.7101449275362319, 'f1': 0.5866037569450568, 'precision': 0.7019327454847817, 'recall': 0.5945093715213148, 'average_precision': np.float64(0.8472452928032411)}
data/test_set/scitail_test.csv
{'best_threshold': 0.9849246231155779, 'accuracy': 0.8015051740357478, 'f1': 0.7933636618244264, 'precision': 0.7923002511086963, 'recall': 0.7945821401351181, 'average_precision': np.float64(0.8097399358207243)}
data/test_set/wic_test.csv
{'best_threshold': 0.9597989949748744, 'accuracy': 0.6142857142857143, 'f1': 0.6140840929138489, 'precision': 0.6145250482641275, 'recall': 0.6142857142857143, 'average_precision': np.float64(0.659639094784424)}
 eval_sts_task
data/test_set/sick_test.csv
Spearman: 0.7482
data/test_set/sts12_test.csv
Spearman: 0.7013
data/test_set/stsb_test.csv
Spearman: 0.7503
=================================================================================================================
FINAL TEST  (pair thresholds calibrated on the test split)
=================================================================================================================
Family         | Benchmark | Primary metric    | Score | Details                                                 
---------------+-----------+-------------------+-------+---------------------------------------------------------
classification | banking77 | f1                | 79.45 | Acc=80.85 F1=79.45                                      
classification | emotion   | f1                | 54.27 | Acc=68.83 F1=54.27                                      
classification | tweet     | f1                | 73.32 | Acc=72.93 F1=73.32                                      
pair           | mrpc      | average_precision | 84.72 | Acc=71.01 F1=58.66 P=70.19 R=59.45 AP=84.72             
pair           | scitail   | average_precision | 80.97 | Acc=80.15 F1=79.34 P=79.23 R=79.46 AP=80.97             
pair           | wic       | average_precision | 65.96 | Acc=61.43 F1=61.41 P=61.45 R=61.43 AP=65.96             
sts            | sick      | spearman          | 74.82 | Spearman=74.82                                          
sts            | sts12     | spearman          | 70.13 | Spearman=70.13                                          
sts            | stsb      | spearman          | 75.03 | Spearman=75.03                                          
---------------+-----------+-------------------+-------+---------------------------------------------------------
summary        | AVG (IOD) | mean              | 65.09 | emotion stsb wic                                        
summary        | AVG (OOD) | mean              | 77.24 | banking77 mrpc scitail sick sts12 tweet                 
summary        | AVG (ALL) | mean              | 73.19 | banking77 emotion mrpc scitail sick sts12 stsb tweet wic
=================================================================================================================
[COMPLETE] pca_mse in 5.1 minutes

########################################################################################
METHOD 3/3: RKD
Log: /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/rkd_train.log
########################################################################################
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
======================================================================
Configuration for RKD method:
======================================================================
  task_type                 : pair_cls
  max_length                : 256
  batch_size                : 64
  epochs                    : 5
  learning_rate             : 2e-05
  min_lr                    : 2e-06
  warmup_ratio              : 0.06
  w_task                    : 1.0
  alpha_dtw                 : 0.5
  w_cls                     : 1.0
  temperature               : 0.1
  student_model_name        : nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base
  teacher_model_name        : Qwen/Qwen3-Embedding-0.6B
  teacher_dtype             : bfloat16
  pooling_method            : last_token
  student_special_token     : ##
  teacher_special_token     : G
  train_data_path           : /content/embedding-kd/data/train_set/train_100k.csv
  cache_dir                 : /content/embedding-kd/runs/teacher_cache
  eval_data_path            : None
  num_workers               : 2
  distill_method            : rkd
  save_dir                  : /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/rkd
  weights_dir               : None
  save_every                : 5
  save_best                 : True
  pair_threshold_source     : test
  eval_retrieval            : False
  debug_align               : False
  evaluate_test_each_epoch  : True
  eval_every                : 0
  seed                      : 42
  w_dist                    : 25.0
  w_angle                   : 50.0
  huber_delta               : 1.0
  normalize_student         : True
  student_pooling           : cls
  eps_norm                  : 1e-12
  cache_teacher             : True
  cache_path                : cache/teacher_train.pt
  normalize_cache           : True
  cache_dtype               : float32
  use_wandb                 : False
======================================================================
Per-epoch evaluation runs on the TEST split and the pair threshold is swept on it. No validation pass will run, so no number this run reports is held out.
Done setup_seed with seed=42
[WARN] Only 1 GPU available -> both on cuda:0
Done setup_devices
Loading tokenizers...
Loading student model: nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base
Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]                                                                                                        
[transformers] [1mBertModel LOAD REPORT[0m from: nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.predictions.decoder.weight             | UNEXPECTED |  | 
Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Loading teacher model: Qwen/Qwen3-Embedding-0.6B
Teacher pooling: last_token (dim 1024)
Student training dtype: torch.float32
Models loaded successfully!
Done setup_models
Loading training data from: /content/embedding-kd/data/train_set/train_100k.csv
Teacher cache (shared directory): /content/embedding-kd/runs/teacher_cache/qwen-qwen3-embedding-0-6b__train_100k__last_token__383e4d6c7e18.pt
Loading cached teacher embeddings from: /content/embedding-kd/runs/teacher_cache/qwen-qwen3-embedding-0-6b__train_100k__last_token__383e4d6c7e18.pt
Loading cached embeddings from: /content/embedding-kd/runs/teacher_cache/qwen-qwen3-embedding-0-6b__train_100k__last_token__383e4d6c7e18.pt
Loaded embeddings: (102361, 1024)
Loaded 102361 cached embeddings (teacher not run for this training)
Teacher model freed from GPU memory
Training samples: 102361
Training batches: 1600
Done setup_data
Checkpoints will be saved to: /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/rkd
Done setup_training
RKD criterion initialized: w_task=1.0, w_dist=25.0, w_angle=50.0, normalize_student=True
============================================================
Starting training...
============================================================
Method: rkd
Student: nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base
Teacher: Qwen/Qwen3-Embedding-0.6B
Epochs: 5
Batch size: 64
Learning rate: 2e-05
============================================================
  self.scheduler.step()
Epoch 1/5:  39%|███▉      | 628/1600 [00:59<00:46, 20.91it/s, avg_loss=4.1851, ms/step=44.7, ms/step(ma)=43.8, it/s=22.83, gpu0=193/4046MB, loss_dist=0.0053, lo
[Epoch 1] Avg step time = 44.54 ms (22.45 it/s)
Done train_epoch 1
[Epoch 1] mean over 102361 examples: loss_total=4.1851  loss_angle=0.0057  loss_dist=0.0063  loss_task=3.7416
============================================================
Evaluation after Epoch 1
============================================================
============================================================
Epoch 2/5:  41%|████      | 649/1600 [00:46<00:44, 21.60it/s, avg_loss=4.1830, ms/step=44.6, ms/step(ma)=44.7, it/s=22.37, gpu0=191/4046MB, loss_dist=0.0047, lo
[Epoch 2] Avg step time = 44.52 ms (22.46 it/s)
Done train_epoch 2
[Epoch 2] mean over 102361 examples: loss_total=4.1831  loss_angle=0.0057  loss_dist=0.0062  loss_task=3.7420
============================================================
Evaluation after Epoch 2
============================================================
============================================================
Epoch 3/5:  82%|████████▏ | 1315/1600 [01:05<00:13, 21.88it/s, avg_loss=4.1841, ms/step=44.5, ms/step(ma)=44.4, it/s=22.54, gpu0=195/4118MB, loss_dist=0.0061, l
[Epoch 3] Avg step time = 44.47 ms (22.49 it/s)
Done train_epoch 3
[Epoch 3] mean over 102361 examples: loss_total=4.1836  loss_angle=0.0057  loss_dist=0.0062  loss_task=3.7423
============================================================
Evaluation after Epoch 3
============================================================
============================================================
Epoch 4/5:  40%|████      | 648/1600 [00:52<00:44, 21.59it/s, avg_loss=4.1857, ms/step=44.5, ms/step(ma)=46.6, it/s=21.46, gpu0=191/4118MB, loss_dist=0.0061, lo
[Epoch 4] Avg step time = 44.54 ms (22.45 it/s)
Done train_epoch 4
[Epoch 4] mean over 102361 examples: loss_total=4.1841  loss_angle=0.0057  loss_dist=0.0062  loss_task=3.7420
============================================================
Evaluation after Epoch 4
============================================================
============================================================
Epoch 5/5:  81%|████████▏ | 1302/1600 [01:09<00:13, 21.71it/s, avg_loss=4.1855, ms/step=44.5, ms/step(ma)=43.9, it/s=22.77, gpu0=190/4118MB, loss_dist=0.0058, l
/content/embedding-kd/src/evaluation/evaluation_automodel.py:295: RuntimeWarning: Published test split banking77_test has normalized-text overlap: train=7, validation=0.
  _validate_classification_pair(train_path, dev_path)
[Epoch 5] Avg step time = 44.53 ms (22.46 it/s)
Done train_epoch 5
[Epoch 5] mean over 102361 examples: loss_total=4.1852  loss_angle=0.0057  loss_dist=0.0062  loss_task=3.7422
============================================================
Evaluation after Epoch 5
============================================================
============================================================
Checkpoint saved: /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/rkd/checkpoint_epoch_5.pt
Done save_checkpoint for epoch 5
Best model saved: /content/embedding-kd/runs/qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-220848/rkd/best_model.pt
============================================================
Training completed!
============================================================
Done train()
Checkpoint for epoch 5 already saved; skipping duplicate.
 eval classifier
data/test_set/banking77_test.csv
banking77_train:   0%|          | 0/36 [00:00<?, ?it/s]                                                                                                         
{'accuracy': 0.6804291287386216, 'f1': 0.6732452459471368}
data/test_set/emotion_test.csv
/content/embedding-kd/src/evaluation/evaluation_automodel.py:295: RuntimeWarning: Published test split tweet_test has normalized-text overlap: train=14, validation=0.
  _validate_classification_pair(train_path, dev_path)
{'accuracy': 0.607754279959718, 'f1': 0.44267549243318466}
data/test_set/tweet_test.csv
{'accuracy': 0.6963869463869464, 'f1': 0.6990641186916857}
 eval_pair_task
data/test_set/mrpc_test.csv
/usr/local/lib/python3.13/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
{'best_threshold': 0.0, 'accuracy': 0.664927536231884, 'f1': 0.3993732590529248, 'precision': 0.332463768115942, 'recall': 0.5, 'average_precision': np.float64(0.7837883408255666)}
data/test_set/scitail_test.csv
{'best_threshold': 0.9899497487437187, 'accuracy': 0.6726246472248354, 'f1': 0.6591969398810444, 'precision': 0.6586408558254281, 'recall': 0.6598793112378922, 'average_precision': np.float64(0.6455666540197154)}
data/test_set/wic_test.csv
{'best_threshold': 0.9899497487437187, 'accuracy': 0.5578571428571428, 'f1': 0.4992369517595182, 'precision': 0.6088045317951021, 'recall': 0.5578571428571428, 'average_precision': np.float64(0.595798867063289)}
 eval_sts_task
data/test_set/sick_test.csv
Spearman: 0.4760
data/test_set/sts12_test.csv
Spearman: 0.2195
data/test_set/stsb_test.csv
Spearman: 0.2208
=================================================================================================================
FINAL TEST  (pair thresholds calibrated on the test split)
=================================================================================================================
Family         | Benchmark | Primary metric    | Score | Details                                                 
---------------+-----------+-------------------+-------+---------------------------------------------------------
classification | banking77 | f1                | 67.32 | Acc=68.04 F1=67.32                                      
classification | emotion   | f1                | 44.27 | Acc=60.78 F1=44.27                                      
classification | tweet     | f1                | 69.91 | Acc=69.64 F1=69.91                                      
pair           | mrpc      | average_precision | 78.38 | Acc=66.49 F1=39.94 P=33.25 R=50.00 AP=78.38             
pair           | scitail   | average_precision | 64.56 | Acc=67.26 F1=65.92 P=65.86 R=65.99 AP=64.56             
pair           | wic       | average_precision | 59.58 | Acc=55.79 F1=49.92 P=60.88 R=55.79 AP=59.58             
sts            | sick      | spearman          | 47.60 | Spearman=47.60                                          
sts            | sts12     | spearman          | 21.95 | Spearman=21.95                                          
sts            | stsb      | spearman          | 22.08 | Spearman=22.08                                          
---------------+-----------+-------------------+-------+---------------------------------------------------------
summary        | AVG (IOD) | mean              | 41.97 | emotion stsb wic                                        
summary        | AVG (OOD) | mean              | 58.29 | banking77 mrpc scitail sick sts12 tweet                 
summary        | AVG (ALL) | mean              | 52.85 | banking77 emotion mrpc scitail sick sts12 stsb tweet wic
=================================================================================================================
[COMPLETE] rkd in 10.1 minutes

Run status:
  geoode   complete               10.7 min
  pca_mse  complete                5.1 min
  rkd      complete               10.1 min

  