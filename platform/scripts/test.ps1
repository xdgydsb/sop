$ErrorActionPreference = "Stop"

$platformRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = @(
    (Join-Path $platformRoot "apps\api"),
    (Join-Path $platformRoot "packages\contracts\src"),
    (Join-Path $platformRoot "services\runtime\src"),
    (Join-Path $platformRoot "services\worker\src")
) -join [IO.Path]::PathSeparator

py -3.10 -m unittest discover `
    -s (Join-Path $platformRoot "tests") `
    -p "test_*.py" `
    -v
