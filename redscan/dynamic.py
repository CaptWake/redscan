import copy
import json
import math
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from redscan.ember import AnalysisError, FileType, detect_file_type

DEFAULT_DYNAMIC_TIMEOUT = 60
NEBULA_MODEL_NAME = "Nebula BPE-50000 Transformer"
PE_FILE_TYPES = {FileType.WIN32, FileType.WIN64, FileType.DOTNET}

OBSERVATION_FIELDS = {
    "apis": ("api", "api_name"),
    "file_access": ("file", "path"),
    "registry_access": ("registry", "path"),
    "network_events.traffic": ("network", "server"),
}

ProgressCallback = Callable[[str], None]
JsonObject = dict[str, Any]


def _notify(progress: ProgressCallback | None, message: str) -> None:
    if progress:
        progress(message)


@dataclass(frozen=True)
class DynamicContribution:
    category: str
    feature: str
    count: int
    contribution: float

    @property
    def direction(self) -> str:
        if self.contribution > 0:
            return "malicious"
        if self.contribution < 0:
            return "benign"
        return "neutral"

    def to_dict(self) -> JsonObject:
        result = asdict(self)
        result["direction"] = self.direction
        return result


@dataclass(frozen=True)
class DynamicGroupContribution:
    name: str
    event_count: int
    contribution: float

    @property
    def direction(self) -> str:
        if self.contribution > 0:
            return "malicious"
        if self.contribution < 0:
            return "benign"
        return "neutral"

    def to_dict(self) -> JsonObject:
        result = asdict(self)
        result["direction"] = self.direction
        return result


@dataclass(frozen=True)
class DynamicPrediction:
    file_path: str
    score: float
    model_name: str
    file_type: FileType
    report_source: str
    token_count: int
    normalized_report: JsonObject
    feature_contributions: tuple[DynamicContribution, ...] = ()
    group_contributions: tuple[DynamicGroupContribution, ...] = ()

    @property
    def classification(self) -> str:
        return "malicious" if self.score >= 0.5 else "benign"


def _create_nebula(config_path: Path | None, timeout: int) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="enable_nested_tensor is True.*",
            category=UserWarning,
        )
        try:
            from nebula import Nebula  # noqa: PLC0415
        except ImportError as exc:
            raise AnalysisError(
                "Nebula backend dependencies are unavailable; run `uv sync`"
            ) from exc

        try:
            model = Nebula(
                vocab_size=50000,
                seq_len=512,
                tokenizer="bpe",
                speakeasy_config=str(config_path) if config_path else None,
            )
        except Exception as exc:
            raise AnalysisError(
                f"Could not initialize the Nebula model: {exc}"
            ) from exc

    try:
        model.dynamic_extractor.speakeasyConfig["timeout"] = timeout
        # Upstream loads the checkpoint in training mode, which leaves dropout active.
        model.model.eval()
        return model
    except Exception as exc:
        raise AnalysisError(f"Could not initialize the Nebula model: {exc}") from exc


def _load_report(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Could not read Speakeasy report {path}: {exc}") from exc


def _normalize_report(model: Any, report: Any) -> JsonObject:
    if isinstance(report, dict) and "entry_points" in report:
        report = report["entry_points"]

    if isinstance(report, list):
        try:
            report = model.dynamic_extractor.filter_and_normalize_report(report)
        except Exception as exc:
            raise AnalysisError(
                f"Could not normalize the Speakeasy report: {exc}"
            ) from exc

    if not isinstance(report, dict):
        raise AnalysisError(
            "Speakeasy report must be a full report, an entry-point list, or a "
            "normalized object"
        )
    if not isinstance(report.get("apis"), list) or not report["apis"]:
        raise AnalysisError("Speakeasy emulation produced no meaningful API sequence")
    return cast(JsonObject, report)


def _predict_score(
    model: Any,
    normalized_report: JsonObject,
) -> tuple[float, NDArray[Any]]:
    try:
        tokens = np.asarray(model.preprocess(normalized_report))
        score = float(model.predict_proba(tokens))
    except Exception as exc:
        raise AnalysisError(f"Nebula prediction failed: {exc}") from exc

    if tokens.size == 0:
        raise AnalysisError("Nebula tokenization produced an empty sequence")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise AnalysisError(f"Nebula returned an invalid probability: {score}")
    return score, tokens


def _records(report: JsonObject, field: str) -> list[JsonObject]:
    records = report.get(field, [])
    if not isinstance(records, list):
        return []
    return [cast(JsonObject, record) for record in records if isinstance(record, dict)]


def _rank_observations(
    report: JsonObject,
    limit: int,
) -> list[tuple[str, str, str, int]]:
    candidates: list[tuple[str, str, str, int]] = []
    for field, (category, value_key) in OBSERVATION_FIELDS.items():
        values = Counter(
            str(record[value_key])
            for record in _records(report, field)
            if record.get(value_key) not in (None, "")
        )
        candidates.extend(
            (field, category, value, count) for value, count in values.items()
        )
    return sorted(candidates, key=lambda item: (-item[3], item[1], item[2]))[:limit]


def _explain(
    model: Any,
    report: JsonObject,
    baseline_score: float,
    top_features: int,
) -> tuple[tuple[DynamicContribution, ...], tuple[DynamicGroupContribution, ...]]:
    group_contributions = []
    for field, (category, _) in OBSERVATION_FIELDS.items():
        records = _records(report, field)
        if not records:
            continue
        ablated = copy.deepcopy(report)
        ablated[field] = []
        if sum(event_counts(ablated).values()) == 0:
            continue
        score, _ = _predict_score(model, ablated)
        group_contributions.append(
            DynamicGroupContribution(category, len(records), baseline_score - score)
        )

    feature_contributions = []
    for field, category, value, count in _rank_observations(report, top_features):
        _, value_key = OBSERVATION_FIELDS[field]
        ablated = copy.deepcopy(report)
        ablated[field] = [
            record
            for record in _records(report, field)
            if str(record.get(value_key)) != value
        ]
        if sum(event_counts(ablated).values()) == 0:
            continue
        score, _ = _predict_score(model, ablated)
        feature_contributions.append(
            DynamicContribution(category, value, count, baseline_score - score)
        )

    return tuple(feature_contributions), tuple(group_contributions)


def event_counts(report: JsonObject) -> dict[str, int]:
    return {
        category: len(_records(report, field))
        for field, (category, _) in OBSERVATION_FIELDS.items()
    }


def save_normalized_report(report: JsonObject, destination: Path) -> None:
    path = destination.expanduser().resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise AnalysisError(
            f"Could not save normalized Speakeasy report {path}: {exc}"
        ) from exc


def predict_dynamic(
    file_path: str | Path,
    *,
    report_path: str | Path | None = None,
    save_report_path: str | Path | None = None,
    config_path: str | Path | None = None,
    timeout: int = DEFAULT_DYNAMIC_TIMEOUT,
    explain: bool = False,
    top_features: int = 10,
    progress: ProgressCallback | None = None,
) -> DynamicPrediction:
    path = Path(file_path).expanduser().resolve()
    file_type = detect_file_type(path)
    if file_type not in PE_FILE_TYPES:
        raise AnalysisError(
            "Nebula dynamic analysis requires a Windows PE file, got "
            f"{file_type.name.lower()}"
        )

    config = Path(config_path).expanduser().resolve() if config_path else None
    if config is not None and not config.is_file():
        raise AnalysisError(f"Speakeasy config does not exist: {config}")
    _notify(progress, "Loading the Nebula model...")
    model = _create_nebula(config, timeout)

    if report_path:
        _notify(progress, "Loading and normalizing the Speakeasy report...")
        source_path = Path(report_path).expanduser().resolve()
        if not source_path.is_file():
            raise AnalysisError(f"Speakeasy report does not exist: {source_path}")
        normalized_report = _normalize_report(model, _load_report(source_path))
        report_source = str(source_path)
    else:
        _notify(progress, "Emulating the sample with Speakeasy...")
        try:
            emulation_report = model.dynamic_analysis_pe_file(str(path))
        except Exception as exc:
            raise AnalysisError(
                f"Speakeasy emulation failed for {path}: {exc}"
            ) from exc
        normalized_report = _normalize_report(model, emulation_report)
        report_source = "live-speakeasy-emulation"

    if save_report_path:
        _notify(progress, "Saving the normalized Speakeasy report...")
        save_normalized_report(normalized_report, Path(save_report_path))

    _notify(progress, "Processing dynamic features with Nebula...")
    score, tokens = _predict_score(model, normalized_report)
    feature_contributions: tuple[DynamicContribution, ...] = ()
    group_contributions: tuple[DynamicGroupContribution, ...] = ()
    if explain:
        _notify(progress, "Calculating dynamic feature impacts...")
        feature_contributions, group_contributions = _explain(
            model,
            normalized_report,
            score,
            top_features,
        )

    pad_token_id = getattr(model.tokenizer, "pad_token_id", 0)
    return DynamicPrediction(
        file_path=str(path),
        score=score,
        model_name=NEBULA_MODEL_NAME,
        file_type=file_type,
        report_source=report_source,
        token_count=int(np.count_nonzero(tokens != pad_token_id)),
        normalized_report=normalized_report,
        feature_contributions=feature_contributions,
        group_contributions=group_contributions,
    )
