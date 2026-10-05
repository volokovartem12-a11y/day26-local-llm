# run.ps1 — локальная LLM через Ollama (день 26). Ключи и интернет-API не нужны.
#
#   .\run.ps1                                   # проверка: сервер, модель, запрос через CLI и 4 запроса через HTTP API
#   .\run.ps1 -Model qwen2.5:3b                 # модель поменьше — быстрее на процессоре
#   .\run.ps1 -Mode ask -Query "Что такое RAG?" # свой вопрос через HTTP API
#   .\run.ps1 -Mode chat                        # диалог прямо в CLI Ollama (выход: /bye)
#
# Если Windows блокирует запуск:  powershell -ExecutionPolicy Bypass -File .\run.ps1

param(
    [ValidateSet("test", "ask", "chat")]
    [string]$Mode = "test",
    [string]$Model = "qwen2.5:7b",
    [string]$Query = ""
)

Set-Location $PSScriptRoot
$api = "http://localhost:11434"

function Find-Python {
    $installed = @(Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe", "$env:ProgramFiles\Python3*\python.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | ForEach-Object { $_.FullName })
    foreach ($candidate in @("py", "python") + $installed) {
        try {
            $output = & $candidate --version 2>&1
            if ($output -match "Python \d") { return $candidate }
        }
        catch { }
    }
    return $null
}

function Find-Ollama {
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @("$env:LOCALAPPDATA\Programs\Ollama\ollama.exe", "$env:ProgramFiles\Ollama\ollama.exe")) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Test-Server {
    try { return (Invoke-RestMethod "$api/api/version" -TimeoutSec 3).version } catch { return $null }
}

# --- 1. Ollama установлена? ---
$ollama = Find-Ollama
if (-not $ollama) {
    Write-Host "Ollama не найдена. Установите её командой:" -ForegroundColor Red
    Write-Host "    winget install Ollama.Ollama" -ForegroundColor Yellow
    Write-Host "затем закройте и заново откройте PowerShell и запустите .\run.ps1 ещё раз."
    exit 1
}
Write-Host "Ollama: $ollama" -ForegroundColor Green

# --- 2. Сервер запущен? ---
$version = Test-Server
if (-not $version) {
    Write-Host "Сервер Ollama не запущен — запускаю в фоне..." -ForegroundColor Cyan
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 30 -and -not $version; $i++) { Start-Sleep -Seconds 1; $version = Test-Server }
    if (-not $version) { Write-Host "Сервер не поднялся за 30 секунд. Запустите приложение Ollama вручную." -ForegroundColor Red; exit 1 }
}
Write-Host "Сервер Ollama работает локально: $api (версия $version)" -ForegroundColor Green

# --- 3. Модель скачана? ---
$names = @((Invoke-RestMethod "$api/api/tags").models | ForEach-Object { $_.name })
if ($names -notcontains $Model -and $names -notcontains "$Model`:latest") {
    Write-Host "`nСкачиваю модель $Model (один раз, несколько гигабайт)..." -ForegroundColor Cyan
    & $ollama pull $Model
    if ($LASTEXITCODE -ne 0) { Write-Host "Не удалось скачать модель $Model." -ForegroundColor Red; exit 1 }
}

if ($Mode -eq "chat") {
    Write-Host "`nДиалог с $Model в CLI Ollama. Выход: /bye`n" -ForegroundColor Cyan
    & $ollama run $Model
    exit
}

$py = Find-Python
if (-not $py) { Write-Host "Python не найден. Установите: winget install Python.Python.3.12" -ForegroundColor Red; exit 1 }

if ($Mode -eq "ask") {
    if (-not $Query) { Write-Host "Укажите -Query ""вопрос""" -ForegroundColor Yellow; exit 1 }
    & $py .\local_llm.py --model $Model --ask $Query
    exit
}

# --- 4. Запрос через CLI ---
Write-Host "`n=== Запрос через CLI: ollama run $Model ===" -ForegroundColor Cyan
Write-Host "Вопрос: Сколько будет 2 + 2? Ответь одним числом."
$cli = & $ollama run $Model "Сколько будет 2 + 2? Ответь одним числом." 2>$null
Write-Host "Ответ модели: $cli" -ForegroundColor Green

# --- 5. Запросы через HTTP API ---
Write-Host "`n=== 4 запроса разной сложности через HTTP API ===" -ForegroundColor Cyan
& $py .\local_llm.py --model $Model

Write-Host "`nГде сейчас работает модель (колонка PROCESSOR: GPU или CPU):" -ForegroundColor Cyan
& $ollama ps
