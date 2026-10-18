param([int]$Port = 7860)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonCandidates = @(
    (Join-Path $repoRoot '.venv\Scripts\python.exe'),
    (Join-Path (Split-Path -Parent $repoRoot) '.venv\Scripts\python.exe')
)
$pythonPath = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $pythonPath) { throw 'Create the Python environment and install requirements.txt first.' }
$url = "http://127.0.0.1:$Port"
try {
    $response = Invoke-WebRequest -Uri "$url/config" -TimeoutSec 2 -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Output "An app is already listening at $url; no second process was started."
        return
    }
} catch { }

$logDirectory = Join-Path $repoRoot '.logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdoutPath = Join-Path $logDirectory "demo-$stamp.stdout.log"
$stderrPath = Join-Path $logDirectory "demo-$stamp.stderr.log"
$appPath = Join-Path $repoRoot 'app.py'
$env:GRADIO_SERVER_PORT = "$Port"
$env:PYTHONUNBUFFERED = '1'
$process = Start-Process -FilePath $pythonPath -ArgumentList @('-u', ('"{0}"' -f $appPath)) `
    -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 500
    $process.Refresh()
    if ($process.HasExited) { throw "The demo exited. Check $stderrPath" }
    try {
        $response = Invoke-WebRequest -Uri "$url/config" -TimeoutSec 2 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Output "Demo running at $url (PID $($process.Id))"
            Write-Output "Logs: $logDirectory"
            return
        }
    } catch { }
}
throw "The demo is still starting. Check $stdoutPath and $stderrPath"
