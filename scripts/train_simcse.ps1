Write-Host "======================================"
Write-Host "Training the SimCSE-only control"
Write-Host "======================================"

$env:CUDA_VISIBLE_DEVICES = "0"
$env:TOKENIZERS_PARALLELISM = "false"

$METHOD = "simcse"
$TRAIN_DATA = "..\data\test_debug.csv"
$STUDENT_MODEL = "..\model_hub\MiniLMv2-L6-H384-distilled-from-BERT-Base\MiniLM-L6-H384-distilled-from-BERT-Base"
$BATCH_SIZE = 32
$EPOCHS = 5
$LR = 2e-5
$MAX_LENGTH = 256
$SIMCSE_VIEW = "dropout"
$SAVE_DIR = "checkpoints/simcse"

# No -teacher_model: this run loads no teacher, it is the control the distilled
# rows are read against.
python ../main.py `
    --method $METHOD `
    --train_data $TRAIN_DATA `
    --student_model $STUDENT_MODEL `
    --batch_size $BATCH_SIZE `
    --epochs $EPOCHS `
    --lr $LR `
    --max_length $MAX_LENGTH `
    --simcse_view $SIMCSE_VIEW `
    --save_dir $SAVE_DIR
