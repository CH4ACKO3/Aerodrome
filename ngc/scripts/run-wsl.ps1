param(
    [ValidateSet("sync", "check", "test", "examples")]
    [string]$Task = "check"
)
$scriptPath = Join-Path $PSScriptRoot "runtime-wsl.sh"
$linuxPath = (& wsl -d Ubuntu -- wslpath -u $scriptPath.Replace('\', '/'))
if ($LASTEXITCODE -ne 0) { throw "Could not resolve the project path in Ubuntu WSL" }
& wsl -d Ubuntu -- bash $linuxPath.Trim() $Task
if ($LASTEXITCODE -ne 0) { throw "Aerodrome task '$Task' failed with exit code $LASTEXITCODE" }
