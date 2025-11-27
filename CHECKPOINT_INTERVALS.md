# Checkpoint Saving at 10% Intervals

## What Changed

The training script (`train.py`) has been modified to automatically save model checkpoints at **10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, and 90% completion** of each epoch.

## Checkpoint Naming

Checkpoints are saved with the following naming convention:
- `checkpoint_epoch_1_10pct.pt` - At 10% of epoch 1
- `checkpoint_epoch_1_20pct.pt` - At 20% of epoch 1
- `checkpoint_epoch_1_30pct.pt` - At 30% of epoch 1
- `checkpoint_epoch_1_40pct.pt` - At 40% of epoch 1
- `checkpoint_epoch_1_50pct.pt` - At 50% of epoch 1
- `checkpoint_epoch_1_60pct.pt` - At 60% of epoch 1
- `checkpoint_epoch_1_70pct.pt` - At 70% of epoch 1
- `checkpoint_epoch_1_80pct.pt` - At 80% of epoch 1
- `checkpoint_epoch_1_90pct.pt` - At 90% of epoch 1
- `checkpoint_epoch_1.pt` - At 100% of epoch 1 (end of epoch, existing behavior)

## What's Saved

Each checkpoint contains:
- Model state (weights)
- Optimizer state
- Scheduler state (if used)
- Current epoch number
- Current batch number
- Total batches in epoch
- Completion percentage
- Current loss value
- Model configuration

## Benefits

1. **More frequent saves**: Don't lose as much progress if training stops
2. **Better recovery**: Can resume from closer to where training stopped
3. **Progress tracking**: See how loss changes throughout the epoch
4. **Flexibility**: Can choose which checkpoint to resume from

## Example Output

During training, you'll see messages like:
```
[CHECKPOINT] Saved at 10% completion: checkpoints_tinystories/checkpoint_epoch_1_10pct.pt (Loss: 2.1234)
[CHECKPOINT] Saved at 20% completion: checkpoints_tinystories/checkpoint_epoch_1_20pct.pt (Loss: 2.0456)
[CHECKPOINT] Saved at 30% completion: checkpoints_tinystories/checkpoint_epoch_1_30pct.pt (Loss: 2.0123)
[CHECKPOINT] Saved at 40% completion: checkpoints_tinystories/checkpoint_epoch_1_40pct.pt (Loss: 1.9876)
[CHECKPOINT] Saved at 50% completion: checkpoints_tinystories/checkpoint_epoch_1_50pct.pt (Loss: 1.9654)
[CHECKPOINT] Saved at 60% completion: checkpoints_tinystories/checkpoint_epoch_1_60pct.pt (Loss: 1.9432)
[CHECKPOINT] Saved at 70% completion: checkpoints_tinystories/checkpoint_epoch_1_70pct.pt (Loss: 1.9234)
[CHECKPOINT] Saved at 80% completion: checkpoints_tinystories/checkpoint_epoch_1_80pct.pt (Loss: 1.9012)
[CHECKPOINT] Saved at 90% completion: checkpoints_tinystories/checkpoint_epoch_1_90pct.pt (Loss: 1.8876)
```

## Resuming from Interval Checkpoints

You can resume from any interval checkpoint using the `--resume` argument:

```bash
python3 train.py --resume checkpoints_tinystories/checkpoint_epoch_1_40pct.pt [other arguments]
```

The training will resume from that exact point in the epoch.

## Storage Considerations

- Each checkpoint saves the full model state, so they can be large
- For a model with ~100M parameters, each checkpoint might be ~400MB
- With 9 interval checkpoints + 1 end-of-epoch checkpoint = 10 checkpoints per epoch
- Consider disk space when training for many epochs
- For 50 epochs: ~500 checkpoints total (if all are kept)

## Disabling Interval Checkpoints

If you want to disable interval checkpoints and only save at the end of each epoch, you can modify the code by commenting out the checkpoint interval saving section in `train_epoch()` function.

## Technical Details

- Checkpoints are saved atomically (write to temp file, then rename)
- GPU operations are synchronized before saving
- Loss value saved is the average loss up to that point in the epoch
- Checkpoints are only saved once per interval (won't duplicate)

