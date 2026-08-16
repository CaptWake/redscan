from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from redscan import __version__
from redscan.dynamic import DynamicPrediction, event_counts
from redscan.ember import EmberPrediction
from redscan.llm import Detection
from redscan.sample import sample_metadata


def ember_report(
    prediction: EmberPrediction,
    explain: bool,
    top_features: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "tool": "redscan",
        "version": __version__,
        "sample": sample_metadata(Path(prediction.file_path), prediction.file_type),
        "backend": "ember",
        "model": prediction.model_name,
        "verdict": {
            "classification": prediction.classification,
            "malicious_probability": prediction.score,
        },
    }
    if explain:
        ranked_features = sorted(
            prediction.feature_contributions,
            key=lambda feature: abs(feature.contribution),
            reverse=True,
        )
        report["explanation"] = {
            "method": "LightGBM TreeSHAP contributions",
            "units": "raw model score (log-odds)",
            "base_value": prediction.base_value,
            "feature_groups": [
                group.to_dict()
                for group in sorted(
                    prediction.group_contributions,
                    key=lambda item: abs(item.contribution),
                    reverse=True,
                )
            ],
            "features": [item.to_dict() for item in ranked_features[:top_features]],
        }
    return report


def llm_report(
    detection: Detection,
    metadata: dict[str, object],
    model: str,
    explain: bool,
    top_features: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "tool": "redscan",
        "version": __version__,
        "sample": metadata,
        "backend": "llm",
        "model": model,
        "verdict": {
            "classification": detection.classification,
            "malicious_probability": detection.malicious_probability,
            "confidence": detection.confidence,
            "summary": detection.summary,
        },
        "limitations": detection.limitations,
    }
    if explain:
        report["explanation"] = {
            "method": "LLM evidence assessment",
            "features": [
                indicator.model_dump()
                for indicator in detection.indicators[:top_features]
            ],
        }
    return report


def dynamic_report(
    prediction: DynamicPrediction,
    explain: bool,
    top_features: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "tool": "redscan",
        "version": __version__,
        "sample": sample_metadata(Path(prediction.file_path), prediction.file_type),
        "backend": "nebula",
        "model": prediction.model_name,
        "verdict": {
            "classification": prediction.classification,
            "malicious_probability": prediction.score,
        },
        "dynamic_analysis": {
            "engine": "Speakeasy",
            "report_source": prediction.report_source,
            "sequence_length": 512,
            "non_padding_tokens": prediction.token_count,
            "event_counts": event_counts(prediction.normalized_report),
        },
        "limitations": [
            "Speakeasy emulates a modeled Windows environment; behavior can "
            "differ on a real host.",
            "The pretrained Nebula model uses only the first 512 BPE tokens "
            "after normalization.",
        ],
    }
    if explain:
        report["explanation"] = {
            "method": "Nebula input-ablation probability delta",
            "units": "malicious probability",
            "note": (
                "Each impact is the baseline score minus the score after "
                "removing that input. "
                "Impacts are local and non-additive."
            ),
            "feature_groups": [
                group.to_dict()
                for group in sorted(
                    prediction.group_contributions,
                    key=lambda item: abs(item.contribution),
                    reverse=True,
                )
            ],
            "features": [
                item.to_dict()
                for item in sorted(
                    prediction.feature_contributions,
                    key=lambda item: abs(item.contribution),
                    reverse=True,
                )[:top_features]
            ],
        }
    return report


def _verdict_style(classification: str) -> str:
    return {
        "malicious": "bold red",
        "suspicious": "bold yellow",
        "benign": "bold green",
    }.get(classification, "bold white")


def _format_bytes(size: int) -> str:
    value = float(size)
    for suffix in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or suffix == "GiB":
            return f"{value:.1f} {suffix}" if suffix != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def render_report(report: dict[str, Any], console: Console) -> None:
    sample = report["sample"]
    verdict = report["verdict"]
    classification = verdict["classification"]
    style = _verdict_style(classification)

    console.print()
    mode = (
        "dynamic malware triage"
        if report["backend"] == "nebula"
        else "static malware triage"
    )
    console.print(f"[bold cyan]REDSCAN[/bold cyan]  [dim]{mode}[/dim]")

    details = Table.grid(padding=(0, 2))
    details.add_column(style="dim", no_wrap=True)
    details.add_column(overflow="fold")
    details.add_row("Sample", Text(sample["name"]))
    details.add_row("SHA256", Text(sample["sha256"]))
    details.add_row("Type", Text(f"{sample['file_type']}  {sample['mime_type']}"))
    details.add_row("Size", _format_bytes(sample["size"]))
    details.add_row("Model", Text(f"{report['backend']} / {report['model']}"))
    dynamic_analysis = report.get("dynamic_analysis")
    if dynamic_analysis:
        total_events = sum(dynamic_analysis["event_counts"].values())
        details.add_row(
            "Behavior",
            Text(
                f"{total_events} events, "
                f"{dynamic_analysis['non_padding_tokens']}/"
                f"{dynamic_analysis['sequence_length']} tokens"
            ),
        )
    console.print(details)

    verdict_text = Text()
    verdict_text.append(classification.upper(), style=style)
    verdict_text.append(
        f"\nMalicious probability: {verdict['malicious_probability'] * 100:.2f}%"
    )
    if "confidence" in verdict:
        verdict_text.append(f"\nModel confidence: {verdict['confidence'] * 100:.2f}%")
    if verdict.get("summary"):
        verdict_text.append(f"\n\n{verdict['summary']}")
    console.print(Panel(verdict_text, title="Verdict", border_style=style.split()[-1]))

    explanation = report.get("explanation")
    if explanation and explanation.get("feature_groups"):
        group_table = Table(title="Feature Group Impact", box=None)
        group_table.add_column("Group", style="cyan")
        group_table.add_column("Direction")
        group_table.add_column("Impact", justify="right")
        for group in explanation["feature_groups"]:
            event_count = group.get("event_count")
            group_name = group["name"]
            if event_count is not None:
                group_name = f"{group_name} ({event_count} events)"
            impact = (
                f"{group['contribution']:+.5g}"
                if report["backend"] == "nebula"
                else f"{group['contribution']:+.5f}"
            )
            group_table.add_row(
                Text(group_name, style="cyan"),
                Text(group["direction"], style=_verdict_style(group["direction"])),
                impact,
            )
        console.print(group_table)

    if explanation and explanation.get("features"):
        feature_table = Table(title="Top Feature Impact", box=None)
        feature_table.add_column("Feature", style="cyan", overflow="fold")
        feature_table.add_column("Direction")
        if report["backend"] == "ember":
            feature_table.add_column("Value", justify="right")
            feature_table.add_column("Impact", justify="right")
            for feature in explanation["features"]:
                feature_table.add_row(
                    Text(feature["name"], style="cyan"),
                    Text(
                        feature["direction"], style=_verdict_style(feature["direction"])
                    ),
                    f"{feature['value']:.5g}",
                    f"{feature['contribution']:+.5f}",
                )
        elif report["backend"] == "nebula":
            feature_table.title = "Dynamic Behavior Impact"
            feature_table.add_column("Count", justify="right")
            feature_table.add_column("Impact", justify="right")
            for feature in explanation["features"]:
                feature_table.add_row(
                    Text(f"{feature['category']}: {feature['feature']}", style="cyan"),
                    Text(
                        feature["direction"], style=_verdict_style(feature["direction"])
                    ),
                    str(feature["count"]),
                    f"{feature['contribution']:+.5g}",
                )
        else:
            feature_table.add_column("Explanation", overflow="fold")
            for feature in explanation["features"]:
                feature_table.add_row(
                    Text(feature["feature"], style="cyan"),
                    Text(feature["impact"], style=_verdict_style(feature["impact"])),
                    Text(feature["explanation"]),
                )
        console.print(feature_table)

    if report.get("limitations"):
        limitations = Text("\n".join(f"- {item}" for item in report["limitations"]))
        console.print(Panel(limitations, title="Limitations", border_style="yellow"))
