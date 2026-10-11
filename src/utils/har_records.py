"""Forecast, coefficient, and edge records shared by the HAR stages."""

import numpy as np

from utils.oos_common import safe_positive_variance


def forecast_record(common, model, predicted_log, floor, fit=None, *, smearing_count=False):
    smearing = fit.smearing_factor if fit else 1.0
    variance, clipped = safe_positive_variance(predicted_log, smearing, floor)
    row = {
        **common, "model": model, "selected_penalty": fit.selected_penalty if fit else np.nan,
        "predicted_log_variance": predicted_log, "predicted_variance": variance, "smearing_factor": smearing,
    }
    if smearing_count:
        row["smearing_clipped_observations"] = fit.smearing_clipped_observations if fit else 0
    row.update(
        prediction_exponent_clipped=clipped, design_condition_number=fit.condition_number if fit else np.nan,
        column_norm_ratio=fit.column_norm_ratio if fit else np.nan,
        lasso_converged=fit.lasso_converged if fit else True,
        lasso_iterations=fit.lasso_iterations if fit else 0,
        lasso_kkt_max_violation=fit.lasso_kkt_max_violation if fit else 0.0,
    )
    return row


def coefficient_records(fit, common, model, labels, *, stock=None, penalty=False):
    values = [fit.intercept, *map(float, fit.own_coefficients)]
    rows = []
    for label, value in zip(("Intercept", *labels), values):
        row = {**common, "model": model}
        if stock is None:
            row["predictor"] = label
        else:
            row.update(predictor_stock=stock, horizon=label)
        row.update(coefficient_original=value, coefficient_standardized=np.nan, selected=True)
        if penalty:
            row["selected_penalty"] = fit.selected_penalty
        rows.append(row)
    return rows


def edge_records(fit, common, model, target, pairs, tolerance, *, signs=False, unit=None, model_first=True):
    rows = []
    for coefficient, standardized, (source, horizon) in zip(
        fit.network_coefficients_original, fit.network_coefficients_standardized, pairs
    ):
        row = dict(common)
        if model_first:
            row["model"] = model
        row.update(source=source, target=target, edge_id=f"{source}->{target}")
        if not model_first:
            row["model"] = model
        row.update({
            "horizon": horizon, "coefficient_original": float(coefficient),
            "coefficient_standardized": float(standardized), "selected": bool(abs(coefficient) > tolerance),
        })
        if signs:
            row["sign"] = int(np.sign(coefficient))
        if unit:
            row["selection_frequency_unit"] = unit
        rows.append(row)
    return rows
