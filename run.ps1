param([string]$Python = "$PSScriptRoot/.venv/Scripts/python.exe")
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (!(Test-Path -LiteralPath $Python)) { throw 'Install requirements into .venv first, or pass -Python with a configured interpreter path.' }
& $Python -m uvicorn projects.api:app --host 127.0.0.1 --port 8011
