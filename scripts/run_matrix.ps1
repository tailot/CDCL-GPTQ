# One-command matrix: cdclq quantize (gptq / cdcl, various bits) + perplexity evaluation.
# Usage:  pwsh -File scripts/run_matrix.ps1
# Reads  : models/TinyLlama-1.1B/model.safetensors, hess-128x2048.safetensors, data/wikitext2-test.txt
# Writes : runs/<name>/{dequant,packed,report.json}, runs/<name>.eval.json, runs/<name>.log
$ErrorActionPreference = 'Continue'
Set-Location "$PSScriptRoot\.."
$exe  = ".\target\release\cdclq.exe"
$py   = ".\.venv\Scripts\python.exe"
$w    = "models\TinyLlama-1.1B\model.safetensors"
$mod  = "models\TinyLlama-1.1B"
$hess = "hess-128x2048.safetensors"
$text = "data\wikitext2-test.txt"
New-Item -ItemType Directory -Force -Path runs | Out-Null

# --- FP16 baseline (no dequant file) ---
if (-not (Test-Path runs\fp16.eval.json)) {
    & $py scripts\eval_ppl.py --model $mod --text $text --seqlen 1024 --json runs\fp16.eval.json *>&1 |
        Out-File -Encoding utf8 runs\fp16.log
    "fp16 done: " + (Get-Content runs\fp16.eval.json -Raw) | Out-File -Append -Encoding utf8 runs\matrix.progress.txt
}

$jobs = @()
foreach ($bits in 4, 3) {
    foreach ($m in "gptq", "cdcl") {
        $jobs += [pscustomobject]@{ name = "b${bits}-${m}"; bits = $bits; method = $m }
    }
}

foreach ($j in $jobs) {
    $log = "runs\$($j.name).log"
    if (Test-Path "runs\$($j.name).eval.json") { continue }
    $t0 = Get-Date
    & $exe quantize --weights $w --hessians $hess --out-dir "runs\$($j.name)" --bits $j.bits --method $j.method --group 128 *>&1 |
        Out-File -Encoding utf8 $log
    "[$((Get-Date) - $t0)] quantize $($j.name) exit=$LASTEXITCODE" | Out-File -Append -Encoding utf8 runs\matrix.progress.txt
    & $py scripts\eval_ppl.py --model $mod --text $text --seqlen 1024 --dequant "runs\$($j.name)\dequant\model.safetensors" --json "runs\$($j.name).eval.json" *>&1 |
        Out-File -Encoding utf8 -Append $log
    "[$((Get-Date) - $t0)] eval $($j.name) -> " + (Get-Content "runs\$($j.name).eval.json" -Raw) | Out-File -Append -Encoding utf8 runs\matrix.progress.txt
}
"ALL DONE" | Out-File -Append -Encoding utf8 runs\matrix.progress.txt
