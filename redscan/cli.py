import argparse
import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.text import Text
from rich_argparse import RichHelpFormatter

from redscan import __version__
from redscan.code_context import (
    REDUMP_BACKENDS,
    REDUMP_OPERATIONS,
)
from redscan.dynamic import DEFAULT_DYNAMIC_TIMEOUT, predict_dynamic
from redscan.ember import AnalysisError, predict_file
from redscan.llm import (
    CONTEXT_STRATEGIES,
    DEFAULT_CONTEXT_STRATEGY,
    DEFAULT_MODEL,
    analyze,
)
from redscan.reporting import dynamic_report, ember_report, llm_report, render_report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _add_result_options(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--explain",
        action="store_true",
        help="Show the evidence behind the verdict",
    )
    command.add_argument(
        "--top-features",
        dest="top_features",
        type=_positive_int,
        default=10,
        metavar="N",
        help="Maximum evidence items to show (default: 10)",
    )
    command.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Write a machine-readable JSON report",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="redscan",
        description="Malware triage through static, dynamic, or LLM analysis",
        formatter_class=RichHelpFormatter,
    )
    commands = parser.add_subparsers(
        dest="analysis",
        title="analysis modes",
        metavar="COMMAND",
        required=True,
    )

    static = commands.add_parser(
        "static",
        help="Score static features with EMBER",
        description="Score a sample using EMBER static features.",
        formatter_class=RichHelpFormatter,
    )
    static.add_argument("sample", type=Path, metavar="FILE", help="Sample to analyze")
    static.add_argument(
        "--model-path",
        type=Path,
        metavar="PATH",
        help="Use a specific LightGBM model instead of automatic selection",
    )
    _add_result_options(static)
    static.set_defaults(handler=_run_static)

    dynamic = commands.add_parser(
        "dynamic",
        help="Emulate Windows behavior with Speakeasy and Nebula",
        description=(
            "Emulate Windows API behavior with Speakeasy and score it with Nebula."
        ),
        formatter_class=RichHelpFormatter,
    )
    dynamic.add_argument("sample", type=Path, metavar="FILE", help="Sample to analyze")
    dynamic.add_argument(
        "--report",
        type=Path,
        metavar="PATH",
        help="Analyze an existing full or normalized Speakeasy report",
    )
    dynamic.add_argument(
        "--save-report",
        type=Path,
        metavar="PATH",
        help="Save the normalized behavior report",
    )
    dynamic.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help="Use a custom Speakeasy JSON configuration",
    )
    dynamic.add_argument(
        "--timeout",
        type=_positive_int,
        default=DEFAULT_DYNAMIC_TIMEOUT,
        metavar="SECONDS",
        help=f"Stop emulation after this duration (default: {DEFAULT_DYNAMIC_TIMEOUT})",
    )
    _add_result_options(dynamic)
    dynamic.set_defaults(handler=_run_dynamic)

    llm = commands.add_parser(
        "llm",
        help="Assess static evidence and extracted code with an LLM",
        description=(
            "Assess static observations and extracted code with an OpenAI model."
        ),
        formatter_class=RichHelpFormatter,
    )
    llm.add_argument("sample", type=Path, metavar="FILE", help="Sample to analyze")
    llm.add_argument(
        "--model",
        default=os.environ.get("REDSCAN_LLM_MODEL", DEFAULT_MODEL),
        metavar="NAME",
        help=(
            "OpenAI model ID; unknown IDs require --context-window "
            "(default: %(default)s)"
        ),
    )
    llm.add_argument(
        "--bin-engine",
        choices=REDUMP_BACKENDS,
        default="auto",
        help="Engine used to extract code for the LLM (default: auto)",
    )
    llm.add_argument(
        "--bin-mode",
        choices=REDUMP_OPERATIONS,
        default="auto",
        help="Code representation sent to the LLM (default: auto)",
    )
    llm.add_argument(
        "--context-strategy",
        choices=CONTEXT_STRATEGIES,
        default=DEFAULT_CONTEXT_STRATEGY,
        help="Handle oversized model input (default: %(default)s)",
    )
    llm.add_argument(
        "--context-window",
        type=_positive_int,
        default=None,
        metavar="TOKENS",
        help="Override the selected model's context window",
    )
    _add_result_options(llm)
    llm.set_defaults(handler=_run_llm)

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def _run_static(args: argparse.Namespace, sample: Path) -> dict[str, Any]:
    with Console(stderr=True).status(
        "[cyan]Processing static features with EMBER...[/cyan]",
        spinner="dots",
    ):
        prediction = predict_file(
            sample,
            model_path=args.model_path,
            explain=args.explain,
        )
    return ember_report(prediction, args.explain, args.top_features)


def _run_dynamic(args: argparse.Namespace, sample: Path) -> dict[str, Any]:
    with Console(stderr=True).status(
        "[cyan]Preparing dynamic analysis...[/cyan]",
        spinner="dots",
    ) as status:
        prediction = predict_dynamic(
            sample,
            report_path=args.report,
            save_report_path=args.save_report,
            config_path=args.config,
            timeout=args.timeout,
            explain=args.explain,
            top_features=args.top_features,
            progress=lambda message: status.update(f"[cyan]{message}[/cyan]"),
        )
    return dynamic_report(prediction, args.explain, args.top_features)


def _run_llm(args: argparse.Namespace, sample: Path) -> dict[str, Any]:
    with Console(stderr=True).status(
        "[cyan]Preparing static LLM analysis...[/cyan]",
        spinner="dots",
    ) as status:
        detection, metadata, _ = analyze(
            sample,
            args.model,
            redump_backend=args.bin_engine,
            redump_operation=args.bin_mode,
            context_strategy=args.context_strategy,
            context_window=args.context_window,
            progress=lambda message: status.update(f"[cyan]{message}[/cyan]"),
        )
    return llm_report(
        detection,
        metadata,
        args.model,
        args.explain,
        args.top_features,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sample = args.sample.expanduser()
    if not sample.is_file():
        parser.error(f"sample does not exist or is not a file: {sample}")

    try:
        report = args.handler(args, sample)
    except AnalysisError as exc:
        Console(stderr=True).print(
            Text.assemble(("redscan:", "bold red"), " ", str(exc))
        )
        return 1

    if args.json_output:
        print(json.dumps(report, indent=2))
    else:
        render_report(report, Console())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
