"""Optional classical next-interval forecaster for training-history learning curves.

MAE here concerns traffic-volume predictions, separately from the anomaly task.
PyTorch is imported lazily so the anomaly project's synthetic tests need no torch.
"""
import copy
import io
import random
import time
import numpy as np
import pandas as pd

WINDOW = 20


def configure_torch(seed):
    """Configure deterministic, single-thread CPU fitting and return torch."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def forecast_arrays(d, training_start_exclusive, fit_end_idx, val_end_idx):
    """Create leakage-safe forecast inputs; raw split indices are exclusive.

    Each target follows 20 past consecutive quarter-hour observations. Training
    windows, including every input and target, lie strictly after the lookback
    start and before fit_end_idx. Scaling uses observed training-period rows only.
    The valid validation/test target sets are independent of training lookback.
    """
    if not 0 < fit_end_idx < val_end_idx < len(d):
        raise ValueError('Need 0 < fit_end_idx < val_end_idx < number of rows')
    t = pd.to_datetime(d['Date'])
    values = pd.to_numeric(d['Volume'], errors='raise').to_numpy(dtype=float)
    if len(d) <= WINDOW or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Need finite nonnegative volumes and enough history')
    if t.isna().any() or not t.is_monotonic_increasing or t.duplicated().any():
        raise ValueError('Timestamps must be ordered, unique and nonmissing')
    start = pd.Timestamp(training_start_exclusive)
    raw_index = np.arange(len(d))
    scaling_mask = (raw_index < fit_end_idx) & (t.to_numpy() > start.to_datetime64())
    if scaling_mask.sum() < WINDOW + 1:
        raise ValueError('Insufficient observations within the training lookback')
    train_values = values[scaling_mask]
    mean = float(train_values.mean())
    std = max(float(train_values.std(ddof=0)), 1.)
    sequences = np.lib.stride_tricks.sliding_window_view(values, WINDOW + 1)
    all_indices = np.arange(WINDOW, len(d))
    interval_ok = np.diff(t.to_numpy()) == np.timedelta64(15, 'm')
    valid = np.lib.stride_tricks.sliding_window_view(interval_ok, WINDOW).all(axis=1)
    indices = all_indices[valid]
    sequences = sequences[valid]
    first_times = t.to_numpy()[indices - WINDOW]
    fit = (indices < fit_end_idx) & (first_times > start.to_datetime64())
    validation = (indices >= fit_end_idx) & (indices < val_end_idx)
    test = indices >= val_end_idx
    if min(fit.sum(), validation.sum(), test.sum()) < 1:
        raise ValueError('Insufficient contiguous windows in a split')
    # Do not tensorise unused pre-lookback targets: their storage would make a
    # small-history fit pay for the complete historical design matrix.
    used = fit | validation | test
    sequences, indices = sequences[used], indices[used]
    fit, validation, test = fit[used], validation[used], test[used]
    return dict(
        x=((sequences[:, :-1] - mean) / std).astype(np.float32)[:, :, None],
        y=((sequences[:, -1] - mean) / std).astype(np.float32),
        indices=indices, actual=sequences[:, -1].copy(), persistence=sequences[:, -2].copy(),
        fit=fit, validation=validation, test=test, train_rows_mask=np.asarray(scaling_mask),
        mean=mean, std=std, training_observations=int(scaling_mask.sum()),
        training_windows=int(fit.sum()), validation_windows=int(validation.sum()), test_windows=int(test.sum()),
    )


def _forecast_metrics(actual, prediction, persistence):
    error = np.asarray(actual) - np.asarray(prediction)
    mae = float(np.mean(abs(error)))
    baseline_mae = float(np.mean(abs(np.asarray(actual) - np.asarray(persistence))))
    total = float(np.sum(actual))
    denominator = float(np.sum((np.asarray(actual) - np.mean(actual)) ** 2))
    return dict(
        mae=mae, rmse=float(np.sqrt(np.mean(error ** 2))),
        wape_pct=100 * float(np.sum(abs(error))) / total if total else float('nan'),
        r2=1 - float(np.sum(error ** 2)) / denominator if denominator else float('nan'),
        persistence_mae=baseline_mae,
        mae_skill_vs_persistence=1 - mae / baseline_mae if baseline_mae else float('nan'),
    )


def run_forecast(d, training_start_exclusive, fit_end_idx, val_end_idx, seed,
                 max_epochs=40, patience=7, on_training_complete=None):
    """Fit direct LSTM(32) and evaluate fixed future targets in count units.

    Configure/import torch before an external RAM sampler if import overhead
    should be excluded from timing. Absolute process RSS still includes libraries.
    training_seconds includes preprocessing, model construction and all training
    epochs with validation used for early stopping. The optional callback fires
    after best-weights restoration and before final validation/test inference.
    """
    if max_epochs < 1 or patience < 1:
        raise ValueError('max_epochs and patience must be positive')
    torch = configure_torch(seed)
    training_started = time.perf_counter()
    a = forecast_arrays(d, training_start_exclusive, fit_end_idx, val_end_idx)
    x = torch.from_numpy(a['x'])
    y = torch.from_numpy(a['y'])
    fit = torch.tensor(a['fit'], dtype=torch.bool)
    validation = torch.tensor(a['validation'], dtype=torch.bool)
    test = torch.tensor(a['test'], dtype=torch.bool)

    class DirectLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent = torch.nn.LSTM(1, 32, batch_first=True)
            self.output = torch.nn.Linear(32, 1)

        def forward(self, batch):
            _, (hidden, _) = self.recurrent(batch)
            return self.output(hidden[-1]).squeeze(-1)

    model = DirectLSTM().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x[fit], y[fit]), batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    vx, vy = x[validation], y[validation]
    preprocessing_seconds = time.perf_counter() - training_started
    fit_started = time.perf_counter()
    best_loss, best_state, best_epoch, stale = float('inf'), None, None, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        summed_loss = 0.
        for bx, by in loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            summed_loss += float(loss.detach()) * len(bx)
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.nn.functional.mse_loss(model(vx), vy))
        history.append(dict(epoch=epoch, training_mse=summed_loss / int(fit.sum()), validation_mse=val_loss))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch, stale = epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    fit_seconds = time.perf_counter() - fit_started
    training_seconds = time.perf_counter() - training_started
    if on_training_complete is not None:
        on_training_complete()
    inference_started = time.perf_counter()
    with torch.no_grad():
        def predict(mask):
            normalized = torch.cat([model(batch) for batch in x[mask].split(512)]).numpy()
            return np.maximum(0., normalized * a['std'] + a['mean'])
        validation_prediction = predict(validation)
        prediction = predict(test)
    inference_seconds = time.perf_counter() - inference_started
    actual = a['actual'][a['test']]
    result = _forecast_metrics(actual, prediction, a['persistence'][a['test']])
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    result.update(
        seed=int(seed), epochs_trained=len(history), best_epoch=int(best_epoch),
        training_observations=a['training_observations'], training_windows=a['training_windows'],
        validation_windows=a['validation_windows'], test_windows=a['test_windows'],
        preprocessing_seconds=preprocessing_seconds, fit_seconds=fit_seconds,
        training_seconds=training_seconds, inference_seconds=inference_seconds,
        validation_mae=float(np.mean(abs(a['actual'][a['validation']] - validation_prediction))),
        scaling_mean=a['mean'], scaling_std=a['std'], model_state_bytes=len(buffer.getvalue()),
        parameter_count=sum(p.numel() for p in model.parameters()),
        test_indices=a['indices'][a['test']].copy(), test_actual=actual.copy(),
        test_prediction=prediction, history=history,
    )
    return result
