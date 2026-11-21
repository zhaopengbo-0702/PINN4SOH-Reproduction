param([string]$Output = 'outputs/full_reproduction/feature_audit')
$ErrorActionPreference = 'Stop'
$destination = [IO.Path]::GetFullPath((Join-Path (Get-Location) $Output))
[IO.Directory]::CreateDirectory($destination) | Out-Null
$sourceRoot = Join-Path $destination 'preprocessing_source'
[IO.Directory]::CreateDirectory($sourceRoot) | Out-Null
$api = 'https://api.github.com/repos/wang-fujin/Battery-dataset-preprocessing-code-library'
$headCommit = Invoke-RestMethod -Uri ($api + '/commits/main')
$tree = Invoke-RestMethod -Uri ($api + '/git/trees/' + $headCommit.sha + '?recursive=1')
$tree | ConvertTo-Json -Depth 9 | Set-Content -Encoding UTF8 (Join-Path $destination 'preprocessing_tree.json')
$records = @()
foreach ($entry in $tree.tree) {
    if ($entry.type -ne 'blob' -or ($entry.path -notmatch '\.(py|md|ipynb)$')) { continue }
    if ($entry.size -gt 3000000) { throw ('Unexpected source size: ' + $entry.path) }
    $blob = Invoke-RestMethod -Uri $entry.url
    $bytes = [Convert]::FromBase64String($blob.content)
    $header = [Text.Encoding]::ASCII.GetBytes('blob ' + $bytes.Length + [char]0)
    $objectBytes = New-Object byte[] ($header.Length + $bytes.Length)
    [Array]::Copy($header, 0, $objectBytes, 0, $header.Length)
    [Array]::Copy($bytes, 0, $objectBytes, $header.Length, $bytes.Length)
    $sha1 = [Security.Cryptography.SHA1]::Create()
    $actual = ([BitConverter]::ToString($sha1.ComputeHash($objectBytes))).Replace('-', '').ToLower()
    $sha1.Dispose()
    if ($actual -ne $entry.sha) { throw ('Git blob identity mismatch: ' + $entry.path) }
    $target = Join-Path $sourceRoot $entry.path
    [IO.File]::WriteAllBytes($target, $bytes)
    $records += [pscustomobject]@{ path=$entry.path; bytes=$bytes.Length; git_blob_sha1=$actual; sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLower(); source_url=('https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library/blob/' + $headCommit.sha + '/' + [Uri]::EscapeDataString($entry.path)) }
}
[pscustomobject]@{ repository='https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library'; retrieved_at=[DateTime]::UtcNow.ToString('o'); commit=$headCommit.sha; commit_date=$headCommit.commit.committer.date; files=$records } | ConvertTo-Json -Depth 7 | Set-Content -Encoding UTF8 (Join-Path $destination 'preprocessing_provenance.json')
$records | Select-Object path,bytes,git_blob_sha1 | Format-Table -AutoSize
