"""Build redump code context for LLM analysis."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from redump import (
    ExtractorError,
    FileFormat,
    Operation,
    available_backends,
    detect_format,
    extract,
)

REDUMP_BACKENDS = ("auto", *available_backends())
REDUMP_OPERATIONS = ("auto", *(operation.value for operation in Operation))


@dataclass(frozen=True)
class RedumpContextBundle:
    observations: dict[str, object]
    code_units: tuple[str, ...]


def _notify(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _select_backend(file_format: FileFormat, requested: str) -> str:
    if requested != "auto":
        return requested
    if file_format is FileFormat.DOTNET:
        return "dncil"
    return "radare2"


def _select_operation(backend: str, requested: str) -> Operation:
    if requested != "auto":
        return Operation(requested)
    if backend == "dncil":
        return Operation.DISASSEMBLE
    return Operation.DECOMPILE


def build_redump_bundle(
    sample: Path,
    *,
    backend: str = "auto",
    operation: str = "auto",
    progress: Callable[[str], None] | None = None,
) -> RedumpContextBundle:
    """Extract code and preserve function boundaries for model context planning."""
    _notify(progress, "Detecting a redump backend...")
    try:
        file_format = detect_format(sample)
        selected_backend = _select_backend(file_format, backend)
        selected_operation = _select_operation(selected_backend, operation)
        result = extract(
            sample,
            backend=selected_backend,
            operation=selected_operation,
            file_format=file_format,
            progress=(
                (lambda message: _notify(progress, f"redump: {message}"))
                if progress is not None
                else None
            ),
        )
    except (ExtractorError, ValueError) as exc:
        return RedumpContextBundle(
            observations={
                "status": "unavailable",
                "requested_backend": backend,
                "requested_operation": operation,
                "error": str(exc),
            },
            code_units=(),
        )

    full_output = result.text
    code_units = tuple(
        replace(result, functions=(function,)).text for function in result.functions
    )
    return RedumpContextBundle(
        observations={
            "status": "ok",
            "backend": result.backend,
            "operation": result.operation.value,
            "file_format": result.file_format.value,
            "function_count": result.function_count,
            "total_characters": len(full_output),
            "included_characters": len(full_output),
            "truncated": False,
            "output": full_output,
        },
        code_units=code_units,
    )


def build_redump_context(
    sample: Path,
    *,
    backend: str = "auto",
    operation: str = "auto",
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Extract code through redump and return bounded, model-ready context."""
    return build_redump_bundle(
        sample,
        backend=backend,
        operation=operation,
        progress=progress,
    ).observations
