$ErrorActionPreference = 'Stop'
$legacyTorchRoot = Join-Path $PSScriptRoot '../outputs/full_reproduction/legacy_runtime'
New-Item -ItemType Directory -Path $legacyTorchRoot -Force | Out-Null
$asset = @{
    filename='torch-1.7.1+cu110-cp37-cp37m-linux_x86_64.whl'
    url='https://download-r2.pytorch.org/whl/cu110/torch-1.7.1%2Bcu110-cp37-cp37m-linux_x86_64.whl'
    sha256='fe553195c62ce2f3f861ae03295c5dc126c7467e005b3cb9dae6c3be84697ab0'
    source_index='https://download.pytorch.org/whl/cu110/torch/'
    kind='wheel'; package='torch'; version='1.7.1+cu110'
}
$destination = Join-Path $legacyTorchRoot $asset.filename
& curl.exe -f -L -sS --connect-timeout 20 --max-time 3600 --retry 3 -C - -o $destination $asset.url
if ($LASTEXITCODE -ne 0) { throw 'Official legacy CUDA torch download failed' }
$actual = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $asset.sha256) { throw 'Legacy CUDA torch SHA256 mismatch' }
$asset.bytes = (Get-Item -LiteralPath $destination).Length
$asset | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $legacyTorchRoot 'legacy_torch_manifest.json') -Encoding utf8
Write-Output "Verified torch 1.7.1 CUDA11.0, $($asset.bytes) bytes."
