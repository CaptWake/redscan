import contextlib
import io
import os
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import lightgbm as lgb
import magic
import numpy as np
import pefile
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from thrember.features import PEFeatureExtractor

from redscan.feature_names import get_feature_metadata


class AnalysisError(RuntimeError):
    """Raised when a detection cannot be completed."""


class FileType(IntEnum):
    UNKNOWN = 0
    WIN32 = 1
    WIN64 = 2
    DOTNET = 3
    APK = 4
    ELF = 5
    PDF = 6


MODEL_MAP = {
    FileType.WIN32: "EMBER2024_Win32.model",
    FileType.WIN64: "EMBER2024_Win64.model",
    FileType.DOTNET: "EMBER2024_Dot_Net.model",
    FileType.APK: "EMBER2024_APK.model",
    FileType.ELF: "EMBER2024_ELF.model",
    FileType.PDF: "EMBER2024_PDF.model",
}
MODEL_REPOSITORY = "joyce8/EMBER2024-benchmark-models"


@dataclass(frozen=True)
class FeatureContribution:
    index: int
    name: str
    group: str
    value: float
    contribution: float

    @property
    def direction(self) -> str:
        if self.contribution > 0:
            return "malicious"
        if self.contribution < 0:
            return "benign"
        return "neutral"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["direction"] = self.direction
        return result


@dataclass(frozen=True)
class GroupContribution:
    name: str
    contribution: float

    @property
    def direction(self) -> str:
        if self.contribution > 0:
            return "malicious"
        if self.contribution < 0:
            return "benign"
        return "neutral"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["direction"] = self.direction
        return result


@dataclass(frozen=True)
class EmberPrediction:
    file_path: str
    score: float
    model_name: str
    file_type: FileType
    base_value: float | None = None
    feature_contributions: tuple[FeatureContribution, ...] = ()
    group_contributions: tuple[GroupContribution, ...] = ()

    @property
    def classification(self) -> str:
        if self.score >= 0.9:
            return "malicious"
        if self.score >= 0.5:
            return "suspicious"
        return "benign"


def model_directory() -> Path:
    configured = os.environ.get("REDSCAN_MODEL_DIR", "~/.redscan/models")
    return Path(configured).expanduser()


def detect_file_type(file_path: str | Path) -> FileType:
    path = Path(file_path)
    try:
        mime_type = magic.from_file(str(path), mime=True)
        with path.open("rb") as handle:
            header = handle.read(1024)

        if mime_type == "application/pdf" or header.startswith(b"%PDF"):
            return FileType.PDF
        if mime_type == "application/zip" and path.suffix.lower() == ".apk":
            return FileType.APK
        if b"AndroidManifest.xml" in header:
            return FileType.APK
        if header.startswith(b"\x7fELF"):
            return FileType.ELF
        if not header.startswith(b"MZ"):
            return FileType.UNKNOWN

        try:
            pe = pefile.PE(str(path), fast_load=True)
            com_descriptor = pe.OPTIONAL_HEADER.DATA_DIRECTORY[14]
            if com_descriptor.VirtualAddress != 0:
                return FileType.DOTNET
            if pe.OPTIONAL_HEADER.Magic == 0x20B:
                return FileType.WIN64
            return FileType.WIN32
        except Exception:
            return FileType.WIN32
    except Exception as exc:
        raise AnalysisError(
            f"Could not detect the file type for {path}: {exc}"
        ) from exc


def resolve_model(
    file_path: str | Path,
    model_path: str | Path | None = None,
) -> tuple[FileType, Path]:
    path = Path(file_path)
    if not path.is_file():
        raise AnalysisError(f"File does not exist: {path}")

    file_type = detect_file_type(path)
    if model_path is not None:
        explicit_path = Path(model_path).expanduser()
        if not explicit_path.is_file():
            raise AnalysisError(f"EMBER model does not exist: {explicit_path}")
        return file_type, explicit_path.resolve()

    model_filename = MODEL_MAP.get(file_type)
    if model_filename is None:
        raise AnalysisError(
            f"Unsupported file type for EMBER: {file_type.name.lower()}"
        )

    destination = model_directory()
    destination.mkdir(parents=True, exist_ok=True)
    cached_model = destination / model_filename
    if cached_model.is_file():
        return file_type, cached_model.resolve()

    try:
        downloaded = hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=model_filename,
            local_dir=str(destination),
        )
    except Exception as exc:
        raise AnalysisError(f"Could not download {model_filename}: {exc}") from exc
    return file_type, Path(downloaded).resolve()


def extract_features(
    file_path: str | Path,
) -> tuple[NDArray[np.float32], PEFeatureExtractor, dict[str, Any]]:
    path = Path(file_path)
    try:
        file_data = path.read_bytes()
    except OSError as exc:
        raise AnalysisError(f"Could not read {path}: {exc}") from exc

    extractor = PEFeatureExtractor()
    try:
        # thrember prints unknown parser warnings to stdout, which would corrupt JSON.
        with contextlib.redirect_stdout(io.StringIO()):
            raw_features = extractor.raw_features(file_data)
        features = extractor.process_raw_features(raw_features).astype(np.float32)
    except Exception as exc:
        raise AnalysisError(
            f"EMBER feature extraction failed for {path}: {exc}"
        ) from exc
    return features, extractor, raw_features


def predict_file(
    file_path: str | Path,
    model_path: str | Path | None = None,
    explain: bool = False,
) -> EmberPrediction:
    path = Path(file_path).expanduser().resolve()
    file_type, resolved_model = resolve_model(path, model_path)
    features, extractor, _ = extract_features(path)

    try:
        model = lgb.Booster(model_file=str(resolved_model))
    except Exception as exc:
        raise AnalysisError(
            f"Could not load EMBER model {resolved_model}: {exc}"
        ) from exc

    if model.num_feature() != features.size:
        raise AnalysisError(
            f"Model expects {model.num_feature()} features, extractor produced "
            f"{features.size}"
        )

    try:
        prediction = np.asarray(model.predict(features.reshape(1, -1))).reshape(-1)
    except Exception as exc:
        raise AnalysisError(f"EMBER prediction failed: {exc}") from exc
    if prediction.size != 1:
        raise AnalysisError("Only binary EMBER models are supported")

    if not explain:
        return EmberPrediction(
            file_path=str(path),
            score=float(prediction[0]),
            model_name=resolved_model.name,
            file_type=file_type,
        )

    try:
        raw_contributions = np.asarray(
            model.predict(features.reshape(1, -1), pred_contrib=True)
        )
    except Exception as exc:
        raise AnalysisError(f"EMBER explanation failed: {exc}") from exc
    if raw_contributions.shape != (1, features.size + 1):
        raise AnalysisError(
            "The selected EMBER model did not return binary feature contributions"
        )

    try:
        labels, groups = get_feature_metadata(extractor)
    except ValueError as exc:
        raise AnalysisError(str(exc)) from exc

    contribution_values = raw_contributions[0, :-1]
    contributions = tuple(
        FeatureContribution(
            index=index,
            name=labels[index],
            group=groups[index],
            value=float(features[index]),
            contribution=float(contribution_values[index]),
        )
        for index in range(features.size)
    )
    group_contributions = tuple(
        GroupContribution(
            name=feature.name,
            contribution=float(
                sum(
                    item.contribution
                    for item in contributions
                    if item.group == feature.name
                )
            ),
        )
        for feature in extractor.features
    )
    return EmberPrediction(
        file_path=str(path),
        score=float(prediction[0]),
        model_name=resolved_model.name,
        file_type=file_type,
        base_value=float(raw_contributions[0, -1]),
        feature_contributions=contributions,
        group_contributions=group_contributions,
    )
