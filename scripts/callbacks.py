import tensorflow as tf
from pathlib import Path

def get_callbacks(config, run_dir, log_dir, checkpoint_dir):
    """
    Creates and returns a list of Keras callbacks based on the config.
    
    Args:
        config (dict): The loaded configuration dictionary.
        run_dir (Path): Path to the current run directory.
        log_dir (Path): Path to the tensorboard logs.
        checkpoint_dir (Path): Path to save model checkpoints.
        
    Returns:
        list: A list of Keras Callback objects.
    """
    cb_config = config['train']['callbacks']
    callbacks = []

    # 1. Early Stopping
    # Stop if validation loss doesn't improve for 'patience' epochs
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=cb_config['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    ))

    # 2. Model Checkpoint
    # Save the best model (overwrite) based on val_loss
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / "best_model.keras"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ))

    # 3. TensorBoard
    # Log metrics for visualization
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(log_dir)))

    # 4. CSV Logger
    # Save text logs for easy plotting without TensorBoard
    callbacks.append(tf.keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")))

    # 5. Learning Rate Scheduler (Optional)
    if cb_config.get('use_lr_scheduler', False):
        scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=cb_config['reduce_lr_factor'],
            patience=cb_config['reduce_lr_patience'],
            min_lr=float(cb_config['reduce_lr_min']),
            verbose=1  # Prints "Reducing learning rate to..."
        )
        callbacks.append(scheduler)
        print(f"✅ Learning Rate Scheduler: ENABLED (Patience: {cb_config['reduce_lr_patience']})")

    return callbacks

if __name__ == "__main__":
    print("This module provides the 'get_callbacks' function.")