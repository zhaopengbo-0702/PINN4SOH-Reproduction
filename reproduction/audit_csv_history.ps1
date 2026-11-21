$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$commit = 'd26bc1f40dc0cdf34405cd5e599a6f6af09742e9'
$directory = Join-Path $workspace "outputs/full_reproduction/upstream_versions/$commit"
$treePath = Join-Path $directory 'git_tree_recursive.json'
$url = "https://api.github.com/repos/wang-fujin/PINN4SOH/git/trees/${commit}?recursive=1"
& curl.exe -f -sS --connect-timeout 15 --max-time 45 --retry 2 --output $treePath $url
if ($LASTEXITCODE -ne 0) { throw 'Official historical tree download failed' }
$tree = Get-Content -LiteralPath $treePath -Raw | ConvertFrom-Json
if ($tree.truncated) { throw 'Historical recursive tree was truncated' }
$repo = Join-Path $workspace 'PINN4SOH'
$old = @($tree.tree | Where-Object { $_.type -eq 'blob' -and $_.path -like 'data/*.csv' })
$currentFiles = @(Get-ChildItem -LiteralPath (Join-Path $repo 'data') -Filter '*.csv' -Recurse -File)
$records = foreach ($entry in $old) {
    $path = Join-Path $repo $entry.path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        [PSCustomObject]@{ path = $entry.path; old_blob_sha1 = $entry.sha; current_blob_sha1 = $null; matches = $false }
        continue
    }
    $bytes = [IO.File]::ReadAllBytes($path)
    $header = [Text.Encoding]::UTF8.GetBytes("blob $($bytes.Length)`0")
    $joined = [byte[]]::new($header.Length + $bytes.Length)
    [Array]::Copy($header, 0, $joined, 0, $header.Length)
    [Array]::Copy($bytes, 0, $joined, $header.Length, $bytes.Length)
    $actual = [Convert]::ToHexString([Security.Cryptography.SHA1]::HashData($joined)).ToLowerInvariant()
    [PSCustomObject]@{ path = $entry.path; old_blob_sha1 = $entry.sha; current_blob_sha1 = $actual; matches = $actual -eq $entry.sha }
}
$extra = @($currentFiles | ForEach-Object { [IO.Path]::GetRelativePath($repo, $_.FullName).Replace('\', '/') } | Where-Object { $_ -notin $old.path })
$report = [PSCustomObject]@{
    historical_commit = $commit
    historical_date = '2024-04-23T11:22:08Z'
    historical_tree_api_url = $url
    historical_tree_file_sha256 = (Get-FileHash -LiteralPath $treePath -Algorithm SHA256).Hash.ToLowerInvariant()
    current_reference_commit = 'cc3cc5053caf38f04e0665f7f88cb109144d035e'
    current_official_archive_sha256 = '9672ecfc35b68c6afb6e596bbc8d5bab33f70eac4f4816d775870078643322e5'
    current_data_source = 'Already downloaded official current archive, checked locally without downloading CSVs again'
    historical_csv_count = $old.Count
    current_csv_count = $currentFiles.Count
    matching_csv_count = @($records | Where-Object matches).Count
    changed_or_missing_csv_count = @($records | Where-Object { -not $_.matches }).Count
    current_extra_paths = $extra
    entries = @($records)
}
$output = Join-Path $workspace 'outputs/full_reproduction/data_csv_history_identity.json'
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $output -Encoding utf8
$report | Select-Object historical_csv_count,current_csv_count,matching_csv_count,changed_or_missing_csv_count,current_extra_paths | ConvertTo-Json -Depth 4
if ($report.changed_or_missing_csv_count -or $extra.Count) { throw 'Data CSV inventory differs between publication-era and current source' }
