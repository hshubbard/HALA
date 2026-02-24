"""
halanalyze — CLI for the HAL Behavior Analyzer.

The CLI is a thin wrapper: it parses arguments and delegates all work to
the core engine. No analysis logic lives here.

Commands
--------
  analyze   Analyse a single trace file.
  diff      Compare two trace files and show regressions/improvements.
  report    Re-render a pre-computed JSON result in a different format.

Examples
--------
  halanalyze analyze trace.json
  halanalyze analyze trace.json --arch ampere --format text
  halanalyze analyze trace.json --severity high
  halanalyze diff old_trace.json new_trace.json
  halanalyze diff old_trace.json new_trace.json --format json
  halanalyze report result.json --format text
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import argparse

# Ensure the package root is importable when running as a script
_here = Path(__file__).resolve().parent.parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from hal_analyzer.core.engine import AnalysisEngine
from hal_analyzer.core.passes.base import Severity


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="halanalyze",
        description="HAL Behavior Analyzer — static analysis of HAL/driver call sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    root.add_argument(
        "--version", action="version", version="halanalyze 1.0.0"
    )

    subs = root.add_subparsers(dest="command", metavar="COMMAND")
    subs.required = True

    # ── analyze ──────────────────────────────────────────────────────
    p_analyze = subs.add_parser(
        "analyze",
        help="Analyse a single HAL trace file",
        description="Parse a trace and run all analysis passes.",
    )
    p_analyze.add_argument("trace", metavar="TRACE", help="Path to trace JSON file")
    p_analyze.add_argument(
        "--arch", "-a", metavar="ARCH",
        help="Target architecture hint (e.g. ampere, rdna3, arm-mali)",
    )
    p_analyze.add_argument(
        "--format", "-f", choices=["text", "json"], default="text",
        dest="output_format",
        help="Output format (default: text)",
    )
    p_analyze.add_argument(
        "--severity", "-s",
        choices=[s.value for s in Severity],
        default=Severity.INFO.value,
        help="Minimum severity to include in output (default: info)",
    )
    p_analyze.add_argument(
        "--output", "-o", metavar="FILE",
        help="Write output to FILE instead of stdout",
    )
    p_analyze.add_argument(
        "--no-colour", action="store_true",
        help="Disable ANSI colour in text output",
    )

    # ── diff ─────────────────────────────────────────────────────────
    p_diff = subs.add_parser(
        "diff",
        help="Compare two trace files",
        description="Analyse both traces and show regressions and improvements.",
    )
    p_diff.add_argument("baseline", metavar="BASELINE", help="Baseline trace JSON")
    p_diff.add_argument("candidate", metavar="CANDIDATE", help="Candidate trace JSON")
    p_diff.add_argument("--arch", "-a", metavar="ARCH", help="Architecture hint")
    p_diff.add_argument(
        "--format", "-f", choices=["text", "json"], default="text",
        dest="output_format",
    )
    p_diff.add_argument("--output", "-o", metavar="FILE")
    p_diff.add_argument("--no-colour", action="store_true")

    # ── report ────────────────────────────────────────────────────────
    p_report = subs.add_parser(
        "report",
        help="Re-render a pre-computed analysis result JSON",
        description="Load a saved analysis JSON and render it in a different format.",
    )
    p_report.add_argument("result", metavar="RESULT", help="Path to analysis result JSON")
    p_report.add_argument(
        "--format", "-f", choices=["text", "json"], default="text",
        dest="output_format",
    )
    p_report.add_argument("--output", "-o", metavar="FILE")
    p_report.add_argument("--no-colour", action="store_true")

    return root


def _write_output(text: str, output_path: str | None) -> None:
    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Output written to: {output_path}", file=sys.stderr)
    else:
        # Reconfigure stdout to UTF-8 on Windows where the default may be cp1252
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass
        sys.stdout.write(text + "\n")
        sys.stdout.flush()


def _cmd_analyze(args: argparse.Namespace, engine: AnalysisEngine) -> int:
    try:
        result = engine.analyze_file(args.trace, arch_hint=args.arch)
    except FileNotFoundError:
        print(f"error: trace file not found: {args.trace}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in trace file: {exc}", file=sys.stderr)
        return 1

    # Filter by severity
    min_sev = Severity(args.severity)
    result.findings = result.by_severity(min_sev)

    if args.output_format == "json":
        text = result.to_json()
    else:
        colour = sys.stdout.isatty() and not args.no_colour
        text = result.to_text_report()   # colour support is always-on for now

    _write_output(text, args.output)

    # Exit code: 1 if any HIGH or CRITICAL findings (useful for CI)
    return 1 if result.high_count() + result.critical_count() > 0 else 0


def _cmd_diff(args: argparse.Namespace, engine: AnalysisEngine) -> int:
    try:
        diff = engine.diff_files(args.baseline, args.candidate, arch_hint=args.arch)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON: {exc}", file=sys.stderr)
        return 1

    if args.output_format == "json":
        text = diff.to_json()
    else:
        text = diff.to_text_report()

    _write_output(text, args.output)
    # Exit 1 if there are regressions
    return 1 if diff.new_findings else 0


def _cmd_report(args: argparse.Namespace, engine: AnalysisEngine) -> int:
    """
    Re-render a pre-computed result JSON.

    The pre-computed JSON was produced by a previous 'analyze --format json'
    run. This command just loads and reformats it — no re-analysis occurs.
    """
    try:
        with open(args.result, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        print(f"error: result file not found: {args.result}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON: {exc}", file=sys.stderr)
        return 1

    if args.output_format == "json":
        # Already JSON — just pretty-print it
        text = json.dumps(data, indent=2)
    else:
        # Reconstruct a minimal text summary from the saved dict
        findings = data.get("findings", [])
        summary = data.get("summary", {})
        lines = [
            "=" * 72,
            "  HAL BEHAVIOR ANALYZER — Saved Analysis Report",
            "=" * 72,
            f"  Source: {data.get('source_path', '(unknown)')}",
            f"  Arch  : {data.get('arch_hint', '(none)')}",
            "",
            f"  Total findings : {summary.get('total_findings', len(findings))}",
            f"  Critical       : {summary.get('critical', 0)}",
            f"  High           : {summary.get('high', 0)}",
            f"  Medium         : {summary.get('medium', 0)}",
            f"  Low            : {summary.get('low', 0)}",
            "",
            "  (Run 'halanalyze analyze <trace>' to recompute with current passes)",
            "=" * 72,
        ]
        for f in findings:
            lines.append(
                f"  [{f.get('severity','?').upper():8s}]  {f.get('title','')}"
            )
            if f.get("related_nodes"):
                lines.append(f"           nodes: {f['related_nodes']}")
            lines.append(f"           {f.get('explanation','')[:120]}")
            lines.append("")
        text = "\n".join(lines)

    _write_output(text, args.output)
    return 0


def main(argv: list | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    engine = AnalysisEngine()

    dispatch = {
        "analyze": _cmd_analyze,
        "diff":    _cmd_diff,
        "report":  _cmd_report,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args, engine)


if __name__ == "__main__":
    sys.exit(main())
