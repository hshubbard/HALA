# HAL Behavior Analyzer (HALA)

A static analysis tool for Hardware Abstraction Layer (HAL) and GPU driver
call sequences. Parses traces, lifts them into a compiler-style IR, and runs
independent analysis passes to detect resource contention, portability
fragility, and performance inefficiencies — without executing any workloads.

---

## Version History

### v1.0 (current)
- **HAL Call Parser** — canonical IR builder with op-table, vendor alias resolution (Vulkan, D3D12, CUDA), and automatic resource lifetime tracking
- **Three analysis passes** — resource contention, fragility / undefined behaviour, performance inefficiency
- **14 checks across 3 passes** — RC-01..04, FR-01..05, PE-01..05
- **CLI** — `analyze`, `diff`, `report` commands; CI-friendly exit codes
- **GUI** — Tkinter frontend; accepts raw trace JSON directly (runs engine in background thread) or pre-computed result JSON
- **Structured JSON output** — machine-readable results for downstream tooling

---

## Quick Start

No installation required. Python 3.10+ standard library only.

```bash
# Analyse a trace (text report to stdout)
python halanalyze.py analyze examples/trace_pingpong.json

# With architecture hint
python halanalyze.py analyze examples/trace_fragile.json --arch ampere

# Export JSON result
python halanalyze.py analyze examples/trace_pingpong.json --format json --output result.json

# Diff two traces
python halanalyze.py diff examples/trace_pingpong.json examples/trace_clean.json

# Launch the GUI (pass a raw trace or a saved result)
python launch_gui.py examples/trace_pingpong.json
```

---

## Project Structure

```
HALA/
├── halanalyze.py               # CLI entry point (no install needed)
├── launch_gui.py               # GUI launcher
├── setup.py                    # pip install support
│
├── hal_analyzer/
│   ├── core/
│   │   ├── ir.py               # IR data structures
│   │   ├── parser.py           # HAL call parser → IR builder
│   │   ├── engine.py           # Public API: AnalysisEngine, AnalysisResult, DiffResult
│   │   ├── reporter.py         # Text and JSON rendering (no analysis logic)
│   │   └── passes/
│   │       ├── base.py                  # AnalysisPass ABC, Finding, Severity
│   │       ├── resource_contention.py   # RC-01..RC-04
│   │       ├── fragility.py             # FR-01..FR-05
│   │       └── performance.py           # PE-01..PE-05
│   │
│   ├── cli/
│   │   └── main.py             # argparse CLI: analyze, diff, report
│   │
│   └── gui/
│       └── app.py              # Tkinter GUI: trace input, call timeline, findings, lifetime bars
│
└── examples/
    ├── trace_pingpong.json     # CPU/GPU ping-pong serialisation + large leak
    ├── trace_fragile.json      # GPU_LOCAL CPU map, missing barriers, arch hints
    ├── trace_perf.json         # Small allocs, map churn, blocking readback
    └── trace_clean.json        # Well-structured reference trace
```

---

## Input Format

Each trace is a JSON file with a `calls` array. Each call maps to one HAL
operation.

```json
{
  "arch": "ampere",
  "metadata": { "app": "my_renderer" },
  "calls": [
    { "op": "ALLOC_BUFFER", "resource": "vbo", "size": 67108864, "memory": "GPU_LOCAL" },
    { "op": "SUBMIT_QUEUE", "queue": "graphics", "reads": ["vbo"] },
    { "op": "WAIT_FENCE",   "resource": "frame_fence", "timeout": "infinite" },
    { "op": "FREE_BUFFER",  "resource": "vbo" }
  ]
}
```

**Field reference:**

| Field        | Type          | Description                                        |
|--------------|---------------|----------------------------------------------------|
| `op`         | string        | Operation name — canonical or vendor alias         |
| `resource`   | string        | Primary resource name (auto-named if omitted)      |
| `size`       | int           | Allocation size in bytes                           |
| `memory`     | string        | `GPU_LOCAL`, `CPU_VISIBLE`, `SHARED`               |
| `cpu_visible`| bool          | Whether CPU can directly access this resource      |
| `queue`      | string        | Target queue (`graphics`, `compute`, `transfer`)   |
| `timeout`    | int or string | Milliseconds, or `"infinite"` / `-1`               |
| `reads`      | list[string]  | Resource names read by this op                     |
| `writes`     | list[string]  | Resource names written by this op                  |
| `blocking`   | string        | `blocking`, `non_blocking`, `conditional`          |
| `arch`       | list[string]  | Architecture hints (e.g. `["ampere"]`)             |

**Vendor aliases** are resolved automatically before parsing:

| Vendor alias | Canonical op |
|---|---|
| `vkAllocateMemory` | `ALLOC_BUFFER` |
| `vkQueueSubmit` | `SUBMIT_QUEUE` |
| `vkWaitForFences` | `WAIT_FENCE` |
| `vkCmdPipelineBarrier` | `PIPELINE_BARRIER` |
| `ID3D12CommandQueue::Signal` | `SIGNAL_FENCE` |
| `cuMemAlloc` | `ALLOC_BUFFER` |
| `cuLaunchKernel` | `DISPATCH` |
| `cuStreamSynchronize` | `FLUSH_QUEUE` |

---

## Intermediate Representation

The parser lifts each raw call into an `IRNode` and builds an `IRTrace`:

```
IRTrace
  nodes        : List[IRNode]              — ordered call sequence
  resources    : Dict[str, Resource]       — all named resources
  lifetimes    : Dict[str, ResourceLifetime] — alloc/free/map intervals
  dependencies : List[Dependency]          — signal→wait ordering edges
  source_arch  : Optional[str]
```

Each `IRNode` carries:

```
IRNode
  index          : int                — position in trace
  op             : str                — canonical UPPER_SNAKE_CASE name
  category       : OpCategory         — ALLOC / FREE / MAP / SUBMIT / SYNC / ...
  blocking       : BlockingKind       — BLOCKING / NON_BLOCKING / CONDITIONAL
  resources_read : List[Resource]
  resources_write: List[Resource]
  sync_point     : bool
  timeout_ms     : Optional[int]      — -1 = infinite
  queue          : Optional[str]
  arch_hints     : Set[str]
```

IR is immutable after construction. All passes read it; none modify it.

---

## Analysis Passes

### Resource Contention Pass

Rules applied independently; all matches are reported.

```
RC-01 — Long-lived exclusive resource ownership
  IF lifetime > 20 ops
  AND at least one SUBMIT_QUEUE occurs during the lifetime
  → severity: MEDIUM (< 50 ops) or HIGH (>= 50 ops)

RC-02 — Serialised SUBMIT→WAIT pattern
  IF SUBMIT_QUEUE is followed within 3 ops by a blocking WAIT
  AND this pattern repeats 2+ times
  → severity: HIGH

RC-03 — Resource never freed in trace
  IF resource is allocated but free_index is None at trace end
  → severity: LOW (small), MEDIUM (> 0 bytes), HIGH (>= 64 MiB)

RC-04 — Queue starvation
  IF multiple queues are referenced
  AND >= 85% of submissions go to one queue
  → severity: MEDIUM
```

### Fragility / Portability Pass

```
FR-01 — Implicit ordering: CPU access after GPU submit without wait
  IF SUBMIT_QUEUE is followed by FREE or MAP of a GPU-written resource
  AND no blocking WAIT intervenes
  → severity: CRITICAL

FR-02 — CPU MAP of GPU_LOCAL memory
  IF MAP_BUFFER targets a resource with memory = GPU_LOCAL
  → severity: HIGH

FR-03 — Missing barrier before resource reuse
  IF a resource is written then read by a GPU op
  AND no PIPELINE_BARRIER or MEMORY_BARRIER intervenes
  → severity: HIGH

FR-04 — Architecture-specific code path
  IF an IRNode carries non-empty arch_hints
  → severity: MEDIUM

FR-05 — Infinite timeout on sync primitive
  IF WAIT_FENCE or WAIT_SEMAPHORE has timeout_ms = -1
  → severity: MEDIUM
```

### Performance Inefficiency Pass

```
PE-01 — Redundant synchronisation
  IF two sync nodes are within 2 ops of each other
  AND no GPU work (DRAW / COMPUTE / TRANSFER / SUBMIT) is between them
  → severity: MEDIUM

PE-02 — Excessive map/unmap churn
  IF a resource has >= 5 map/unmap cycles in its lifetime
  → severity: MEDIUM (< 10 cycles) or HIGH (>= 10 cycles)

PE-03 — Small repeated allocations
  IF >= 4 allocations are each < 256 KiB
  → severity: MEDIUM

PE-04 — Blocking readback in hot path
  IF a blocking READBACK or QUERY appears between two SUBMIT_QUEUE calls
  → severity: HIGH

PE-05 — Staging buffer persistently mapped across submission
  IF a CPU_VISIBLE resource is mapped and not unmapped before a SUBMIT_QUEUE
  → severity: LOW
```

---

## CLI Reference

```
python halanalyze.py analyze TRACE [--arch ARCH]
                                   [--format text|json]
                                   [--severity info|low|medium|high|critical]
                                   [--output FILE]

python halanalyze.py diff BASELINE CANDIDATE [--arch ARCH]
                                             [--format text|json]
                                             [--output FILE]

python halanalyze.py report RESULT [--format text|json] [--output FILE]
```

**Exit codes:**
- `0` — no HIGH or CRITICAL findings
- `1` — HIGH or CRITICAL findings present, or parse/IO error

The exit code makes `halanalyze analyze` drop-in usable in CI pipelines.

---

## Outputs

| Path | Contents |
|------|----------|
| stdout (text) | Human-readable report grouped by severity and category |
| `--format json` | Full structured result: summary, findings, IR trace, lifetimes |
| `diff` text | Regressions and improvements between two traces |
| `diff --format json` | Machine-readable diff with new/fixed finding lists |

**JSON result structure:**

```json
{
  "source_path": "trace.json",
  "arch_hint": "ampere",
  "summary": { "total_findings": 8, "critical": 0, "high": 2, "medium": 3, "low": 3, "info": 0 },
  "findings": [ { "pass": "...", "category": "...", "severity": "...", "title": "...",
                  "explanation": "...", "suggestion": "...", "related_nodes": [...] } ],
  "trace": { "nodes": [...], "resources": {...}, "lifetimes": {...}, "dependencies": [...] }
}
```

---

## Extending the Tool

### Adding a new analysis pass

Create `hal_analyzer/core/passes/my_pass.py`:

```python
from .base import AnalysisPass, Finding, FindingCategory, Severity
from ..ir import IRTrace

class MyPass(AnalysisPass):
    def run(self, trace: IRTrace) -> list[Finding]:
        findings = []
        for node in trace.nodes:
            if <condition>:
                findings.append(Finding(
                    pass_name=self.name,
                    category=FindingCategory.CORRECTNESS,
                    severity=Severity.HIGH,
                    title="Short title",
                    explanation="Full explanation...",
                    related_nodes=[node.index],
                    suggestion="How to fix...",
                ))
        return findings
```

Add it to `DEFAULT_PASSES` in `hal_analyzer/core/engine.py`. No other changes required.

### Adding a new HAL backend

Subclass `HALParser` and override `_pre_process()`:

```python
from hal_analyzer.core.parser import HALParser

class MetalParser(HALParser):
    _MAP = {
        "MTLCommandBuffer.commit":          "SUBMIT_QUEUE",
        "MTLCommandBuffer.waitUntilCompleted": "FLUSH_QUEUE",
        "MTLBuffer.contents":               "MAP_BUFFER",
    }
    def _pre_process(self, raw_calls):
        for call in raw_calls:
            call["op"] = self._MAP.get(call.get("op", ""), call.get("op", ""))
        return raw_calls
```

Pass it to the engine: `AnalysisEngine(parser=MetalParser())`.

---

## Limitations

**Static analysis only.** The tool reasons about call order and resource
names as declared in the trace. It does not execute code, simulate the
GPU pipeline, or observe actual memory traffic.

**Trace fidelity.** Analysis quality is bounded by how accurately the
input trace reflects the real call sequence. Omitted calls (e.g. implicit
driver barriers) will cause false positives in FR-03.

**Heuristic thresholds.** All numeric thresholds (lifetime length,
map-churn count, small-alloc size) are documented heuristics, not
hardware-calibrated values. They can be tuned in the pass source files.

**No intra-frame state.** The tool treats each trace as a self-contained
unit. Resources that are legitimately kept alive across frames will be
flagged as RC-03 (never freed). Provide per-frame traces or annotate
long-lived resources explicitly.

**Vendor alias coverage.** The built-in alias table covers common Vulkan,
D3D12, and CUDA entry points. Uncommon or extension entry points will
pass through as `OTHER` category ops and will not trigger most checks.

---

## Why This Approach Is Valid

1. **Compiler-style architecture.** IR is immutable; passes are stateless
   and independently testable. A failing pass cannot corrupt another pass
   or the IR.

2. **Traceable findings.** Every `Finding` carries `related_nodes` — the
   exact IR node indices that triggered it. Nothing is inferred globally
   without a concrete source location.

3. **Documented rules.** Each check has a named identifier (RC-01, FR-03,
   etc.), a stated condition, and a stated severity. There are no hidden
   scoring formulas.

4. **Honest limitations.** The Limitations section documents exactly what
   static trace analysis cannot see.

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Uses `match`-free dataclasses and `str \| Path` unions |
| `tkinter` | GUI only — included with standard CPython on all platforms |

No third-party packages required.
