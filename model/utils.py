import os


def latest_checkpoint(checkpoint_dir):

    checkpoints = sorted(os.listdir(checkpoint_dir))
    if len(checkpoints) == 0:
        return None
    return os.path.join(checkpoint_dir, checkpoints[-1])