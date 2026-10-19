$ErrorActionPreference = "Stop"

if (-not $env:OWNER) { throw "请设置 OWNER,例如 `$env:OWNER='Cxx-hyy'" }
if (-not $env:REPO)  { throw "请设置 REPO,例如 `$env:REPO='badminton-prediction'" }

$jsonPath = Join-Path $PSScriptRoot "branch_protection.json"
if (-not (Test-Path $jsonPath)) {
    throw "找不到分支保护规则文件: $jsonPath"
}

$body = Get-Content $jsonPath -Raw

foreach ($branch in @("dev", "staging", "main")) {
    Write-Host "配置分支保护: $branch"

    $tmpPath = [System.IO.Path]::GetTempFileName()
    try {
        [System.IO.File]::WriteAllText(
            $tmpPath,
            $body,
            [System.Text.UTF8Encoding]::new($false)
        )

        gh api --method PUT `
            "repos/$env:OWNER/$env:REPO/branches/$branch/protection" `
            --input $tmpPath

        if ($LASTEXITCODE -ne 0) {
            throw "配置 $branch 分支保护失败"
        }
    }
    finally {
        Remove-Item $tmpPath -Force -ErrorAction SilentlyContinue
    }
}
Write-Host "全部分支保护配置完成."