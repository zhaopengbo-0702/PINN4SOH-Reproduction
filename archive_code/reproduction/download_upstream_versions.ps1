$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$destinationRoot = Join-Path $workspace 'outputs/full_reproduction/upstream_versions'
$commits = @('d26bc1f40dc0cdf34405cd5e599a6f6af09742e9', 'cc3cc5053caf38f04e0665f7f88cb109144d035e')
$wanted = @('README.md', 'fix.md', 'Model/Model.py', 'main_XJTU.py', 'main_TJU.py', 'main_MIT.py', 'main_HUST.py', 'main_comparision.py', 'main_adaptation - fine-tuning.py', 'dataloader/dataloader.py', 'utils/util.py')
foreach ($commit in $commits) {
    $folder = Join-Path $destinationRoot $commit
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    $commitMetadata = (& curl.exe -f -sS --connect-timeout 15 --max-time 45 --retry 2 "https://api.github.com/repos/wang-fujin/PINN4SOH/commits/$commit") | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or $commitMetadata.sha -ne $commit) { throw 'Commit verification failed' }
    $tree = (& curl.exe -f -sS --connect-timeout 15 --max-time 45 --retry 2 "https://api.github.com/repos/wang-fujin/PINN4SOH/git/trees/${commit}?recursive=1") | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or $tree.truncated) { throw 'Tree retrieval failed' }
    $entries = $tree.tree | Where-Object { $_.type -eq 'blob' -and $_.path -in $wanted }
    $records = $entries | ForEach-Object -Parallel {
        $entry = $_
        $target = Join-Path $using:folder $entry.path
        New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent) | Out-Null
        $blob = (& curl.exe -f -sS --connect-timeout 15 --max-time 45 --retry 2 $entry.url) | ConvertFrom-Json
        if ($LASTEXITCODE -ne 0 -or $blob.sha -ne $entry.sha) { throw "Blob metadata failed: $($entry.path)" }
        $bytes = [Convert]::FromBase64String($blob.content)
        $header = [Text.Encoding]::UTF8.GetBytes("blob $($bytes.Length)`0")
        $joined = [byte[]]::new($header.Length + $bytes.Length)
        [Array]::Copy($header, 0, $joined, 0, $header.Length)
        [Array]::Copy($bytes, 0, $joined, $header.Length, $bytes.Length)
        $gitSha = [Convert]::ToHexString([Security.Cryptography.SHA1]::HashData($joined)).ToLowerInvariant()
        if ($gitSha -ne $entry.sha) { throw "Git blob hash mismatch: $($entry.path)" }
        [IO.File]::WriteAllBytes($target, $bytes)
        [PSCustomObject]@{
            path = $entry.path
            blob_sha1 = $gitSha
            sha256 = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
            bytes = $bytes.Length
            api_url = $entry.url
            source_url = "https://github.com/wang-fujin/PINN4SOH/blob/$($using:commit)/$($entry.path)"
        }
    } -ThrottleLimit 3
    $report = [PSCustomObject]@{
        commit = $commit
        date = $commitMetadata.commit.author.date
        message = $commitMetadata.commit.message
        commit_api_url = $commitMetadata.url
        tree_sha = $commitMetadata.commit.tree.sha
        files = @($records | Sort-Object path)
        absent_paths = @($wanted | Where-Object { $_ -notin $entries.path })
        verification = 'Official GitHub HTTPS API with certificate verification; decoded bytes match Git blob SHA1'
        execution = 'Historical source downloaded for audit only; not executed'
    }
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $folder 'snapshot_metadata.json') -Encoding utf8
    $report | Select-Object commit,date,absent_paths | ConvertTo-Json -Depth 4
}
