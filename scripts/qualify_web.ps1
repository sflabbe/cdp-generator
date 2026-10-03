# Run as a script (not pasted line-by-line) so any failed native command stops.
[CmdletBinding()]
param([switch]$CommitAndPush)

$ErrorActionPreference = "Stop"
$previousUtf8 = $env:PYTHONUTF8
$repoRoot = Split-Path -Parent $PSScriptRoot

function Invoke-Checked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Command"
    }
}

Push-Location $repoRoot
try {
    # Frozen tests use Python's default text encoding; keep them byte-identical.
    # Environment mode also propagates UTF-8 to the workbook test subprocess.
    $env:PYTHONUTF8 = "1"
    Invoke-Checked { uv run --locked --extra web python scripts/normalize_frozen_text.py }
    Invoke-Checked { git diff --check }
    Invoke-Checked { uv run --locked --extra web ruff check . }
    Invoke-Checked { uv run --locked --extra web mypy cdp_generator }
    Invoke-Checked { uv run --locked --extra web python -m pytest -q }

    if ($CommitAndPush) {
        Invoke-Checked { git add -A }
        Invoke-Checked { git diff --cached --check }
        Invoke-Checked { git diff --cached --stat }
        Invoke-Checked { git commit -m "Fix Windows qualification encoding and frozen LF checkouts" }
        Invoke-Checked { git push }
        Invoke-Checked { git rev-parse HEAD }
        Invoke-Checked { git status --short }
    }
} finally {
    $env:PYTHONUTF8 = $previousUtf8
    Pop-Location
}
