"""Train and evaluate a binary EEG classifier."""

import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torchmetrics.classification import BinaryF1Score
from tqdm.auto import tqdm


class EarlyStopping:
    """Stop training when validation loss does not improve."""

    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best is None or val_loss < self.best:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True


class TrainerClassify:
    """Run binary training, validation, and testing."""

    def __init__(
        self,
        model,
        epochs,
        optimizer,
        scheduler,
        loss_fn,
        es_patience,
        batch_size,
        directory,
        device="cpu",
        early_stopping_enabled=False,
    ):
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.es_patience = es_patience
        self.batch_size = batch_size
        self.directory = directory
        self.device = device
        self.early_stopping_enabled = early_stopping_enabled


        self.binary_f1 = BinaryF1Score().to(self.device)
        self.tracker = {"train_tracker": [], "val_tracker": []}

    def _save_progress_checkpoint(self, epoch_idx: int):
        """Save intermediate checkpoint for long runs."""
        base, ext = os.path.splitext(self.directory)
        ext = ext or ".pt"
        ckpt_path = f"{base}.epoch{epoch_idx}{ext}"
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"checkpoint saved: {ckpt_path}")

    def _process_batch(self, data, target, training=True):
        data, target = data.to(self.device), target.to(self.device)

        output = self.model(data)
        if output.ndim != 2 or output.shape[1] != 2:
            raise ValueError(f"Expected logits [N, 2], got {tuple(output.shape)}")

        loss = self.loss_fn(output, target)
        probs_pos = torch.softmax(output, dim=1)[:, 1]
        f1 = self.binary_f1(probs_pos, target.int())

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, f1, output

    def train_classify(self, train_loader, val_loader):
        """Train model and track train/val losses."""
        train_loss_tracker, val_loss_tracker = [], []

        early_stopping = None
        if self.early_stopping_enabled:
            early_stopping = EarlyStopping(patience=self.es_patience)

        checkpoint_gap = max(1, self.epochs // 5)

        for epoch in tqdm(range(self.epochs)):
            t0 = time.time()
            batch_train_loss, batch_train_f1 = 0.0, 0.0
            batch_val_loss, batch_val_f1 = 0.0, 0.0

            self.model.train()
            for data, target in train_loader:
                loss, score, _ = self._process_batch(data, target, training=True)
                batch_train_loss += loss.item()
                batch_train_f1 += float(score.item())

            final_train_loss = batch_train_loss / max(len(train_loader), 1)
            final_train_f1 = batch_train_f1 / max(len(train_loader), 1)

            self.model.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    loss, score, _ = self._process_batch(data, target, training=False)
                    batch_val_loss += loss.item()
                    batch_val_f1 += float(score.item())

            final_val_loss = batch_val_loss / max(len(val_loader), 1)
            final_val_f1 = batch_val_f1 / max(len(val_loader), 1)
            self.scheduler.step(final_val_loss)

            print(
                f"Epoch: {epoch + 1}/{self.epochs}, "
                f"Train Loss: {final_train_loss:.4f}, Train F1: {final_train_f1:.4f}, "
                f"Val Loss: {final_val_loss:.4f}, Val F1: {final_val_f1:.4f}, "
                f"Time: {time.time() - t0:.2f} sec"
            )

            train_loss_tracker.append(final_train_loss)
            val_loss_tracker.append(final_val_loss)

            epoch_num = epoch + 1
            if epoch_num % checkpoint_gap == 0 or epoch_num == self.epochs:
                self._save_progress_checkpoint(epoch_num)

            if early_stopping is not None:
                early_stopping(final_val_loss)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        print(f"saving to {self.directory}")
        torch.save(self.model.state_dict(), self.directory)
        self.tracker["train_tracker"] = train_loss_tracker
        self.tracker["val_tracker"] = val_loss_tracker
        return self.tracker

    def eval_classify(self, test_dataloader):
        """Evaluate model and return losses, f1, predictions, and logits."""
        model_path = self.directory
        print(f"loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        batch_test_loss, batch_test_f1 = 0.0, 0.0
        y_pred = []
        y_pred_prob = []

        with torch.no_grad():
            for data, target in test_dataloader:
                loss, score, output = self._process_batch(data, target, training=False)
                batch_test_loss += loss.item()
                batch_test_f1 += float(score.item())
                y_pred.extend(np.argmax(output.cpu().numpy(), axis=1))
                y_pred_prob.extend(output.cpu().numpy())

        final_test_loss = batch_test_loss / max(len(test_dataloader), 1)
        final_test_f1 = batch_test_f1 / max(len(test_dataloader), 1)
        return final_test_loss, final_test_f1, y_pred, np.array(y_pred_prob)

    def eval_result(self, final_test_loss, final_test_f1, y_pred, y_true, y_pred_prob=None):
        """Compute and return binary classification metrics."""
        y_pred_argm = np.array(y_pred)

        accuracy = accuracy_score(y_true, y_pred_argm)
        sensitivity, specificity, f1_binary, f1_macro, f1_weighted, bal_acc, auc_score, cohen_kappa = (
            self.sen_spec_f1_metrics(y_true, y_pred_argm, y_pred_prob)
        )
        print(f"Binary F1: {final_test_f1:.4f}, Macro F1: {f1_macro:.4f}")

        evaluation = {
            "testing_loss": final_test_loss,
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1-score-binary": f1_binary,
            "f1-score-macro": f1_macro,
            "f1-score-weighted": f1_weighted,
            "balanced_accuracy_score": bal_acc,
            "roc_auc_score": auc_score,
            "cohen_kappa_score": cohen_kappa,
        }
        return {"y_true": y_true, "y_pred": y_pred_argm}, evaluation

    def sen_spec_f1_metrics(self, y_true, y_pred, y_pred_prob=None):
        """Compute sensitivity, specificity, F1, AUC and kappa for binary labels."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        recall = recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        f1_score_binary = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        sk_balanced_accuracy_score = balanced_accuracy_score(y_true, y_pred)

        if y_pred_prob is not None and y_pred_prob.ndim == 2 and y_pred_prob.shape[1] == 2 and len(set(y_true)) == 2:
            auc_score = roc_auc_score(y_true, y_pred_prob[:, 1])
        else:
            auc_score = None

        f1_score_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_score_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cohen_kappa = cohen_kappa_score(y_true, y_pred)

        print(f"Verifying sensitivity {sensitivity} and recall {recall}")
        return (
            sensitivity,
            specificity,
            f1_score_binary,
            f1_score_macro,
            f1_score_weighted,
            sk_balanced_accuracy_score,
            auc_score,
            cohen_kappa,
        )
