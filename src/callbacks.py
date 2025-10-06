import logging
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os
import time

class TrainingLoggerCallback(TrainerCallback):
    """
    Custom callback for enhanced logging during training.
    Logs training progress, metrics, and timing information.
    """

    def __init__(self, log_every_n_steps=50):
        self.log_every_n_steps = log_every_n_steps
        self.start_time = None
        self.last_log_time = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when training begins."""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        logging.info("=" * 50)
        logging.info("TRAINING STARTED")
        logging.info(f"Model: {args.output_dir}")
        logging.info(f"Total training steps: {state.max_steps}")
        logging.info(f"Batch size: {args.per_device_train_batch_size}")
        logging.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        logging.info(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        logging.info("=" * 50)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each training step."""
        if state.global_step % self.log_every_n_steps == 0:
            current_time = time.time()
            elapsed = current_time - self.start_time
            step_elapsed = current_time - self.last_log_time

            # Calculate ETA
            if state.global_step > 0:
                avg_time_per_step = elapsed / state.global_step
                remaining_steps = state.max_steps - state.global_step
                eta = remaining_steps * avg_time_per_step
                eta_str = f"{eta / 3600:.1f}h" if eta > 3600 else f"{eta / 60:.1f}m"
            else:
                eta_str = "N/A"

            logging.info(
                f"Step {state.global_step}/{state.max_steps} | "
                f"Loss: {state.log_history[-1].get('train_loss', 'N/A'):.4f} | "
                f"LR: {state.log_history[-1].get('learning_rate', 'N/A'):.2e} | "
                f"Elapsed: {elapsed/60:.1f}m | "
                f"Step time: {step_elapsed:.2f}s | "
                f"ETA: {eta_str}"
            )

            self.last_log_time = current_time

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics:
            eval_loss = metrics.get('eval_loss', 'N/A')
            logging.info("=" * 30)
            logging.info("EVALUATION RESULTS")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}" if isinstance(value, (int, float)) else f"{key}: {value}")
            logging.info("=" * 30)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when a checkpoint is saved."""
        checkpoint_dir = f"{args.output_dir}/{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        logging.info(f"Checkpoint saved to: {checkpoint_dir}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when training ends."""
        total_time = time.time() - self.start_time
        logging.info("=" * 50)
        logging.info("TRAINING COMPLETED")
        logging.info(f"Total training time: {total_time/60:.1f} minutes")
        logging.info(f"Final step: {state.global_step}")
        logging.info(f"Best metric: {getattr(state, 'best_metric', 'N/A')}")
        logging.info("=" * 50)


class EarlyStoppingCallback(TrainerCallback):
    """
    Custom early stopping callback based on evaluation loss.
    """

    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Check for early stopping."""
        if metrics and 'eval_loss' in metrics:
            current_loss = metrics['eval_loss']
            if current_loss < self.best_loss - self.min_delta:
                self.best_loss = current_loss
                self.counter = 0
            else:
                self.counter += 1
                logging.info(f"Early stopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                logging.info("Early stopping triggered!")
                control.should_training_stop = True