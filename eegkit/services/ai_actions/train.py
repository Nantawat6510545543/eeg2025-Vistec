"""Train a simple classifier on epochs and return the model."""

from __future__ import annotations

from . import register_ai
from .helpers import encode_labels, select_device, model_factory, make_dataloader
from ...models.dtos import AITrainParamsDTO, BaseTaskDTO


@register_ai("Train", AITrainParamsDTO)
def train(self, task_dto: BaseTaskDTO, params: AITrainParamsDTO):
    """Train a simple classifier and return the in-memory model (no saving)."""
    try:
        import torch  # local import to avoid hard dependency for non-AI flows
    except Exception:
        return {"status": "error", "reason": "torch not available"}

    X, y_raw, meta = self._build_epoch_dataset(task_dto, params)
    if X is None:
        return {"status": "unavailable", "reason": meta.get("reason")}

    y_idx, classes = encode_labels(y_raw)

    device = select_device(
        (params.device or ["auto"])[0] if isinstance(params.device, list) else params.device
    )
    n_e, n_c, n_t = X.shape
    model, info = model_factory(
        (params.model or [None])[0] if isinstance(params.model, list) else params.model,
        n_c,
        n_t,
        len(classes),
    )
    if model is None:
        return {"status": "unsupported", **info}
    model = model.to(device)
    adapter = info.get("input_adapter")

    dl = make_dataloader(X, y_idx, int(params.batch_size))
    criterion = __import__("torch").nn.CrossEntropyLoss()
    optimizer = __import__("torch").optim.Adam(model.parameters(), lr=float(params.lr))
    model.train()
    for _ in range(int(params.epochs_n)):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            if adapter == "flatten":
                xb_in = xb.reshape(xb.size(0), -1)
            elif adapter == "channels_times":
                xb_in = xb
            else:
                xb_in = xb
            out = model(xb_in)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    return model
