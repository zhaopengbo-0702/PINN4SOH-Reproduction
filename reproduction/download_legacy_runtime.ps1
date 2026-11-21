$ErrorActionPreference = 'Stop'
$legacyRoot = Join-Path $PSScriptRoot '../outputs/full_reproduction/legacy_runtime'
New-Item -ItemType Directory -Force -Path $legacyRoot | Out-Null
$condaPackages = @(
    @{ Name='python'; Version='3.7.10'; Build='h12debd9_4' },
    @{ Name='openssl'; Version='1.1.1w' },
    @{ Name='libffi'; Version='3.3' },
    @{ Name='ncurses'; Version='6.2' },
    @{ Name='readline'; Version='8.1' },
    @{ Name='sqlite'; Version='3.35.4' },
    @{ Name='tk'; Version='8.6.10' },
    @{ Name='xz'; Version='5.2.5' },
    @{ Name='zlib'; Version='1.2.11' },
    @{ Name='libgcc-ng'; Version='9.3.0' },
    @{ Name='libstdcxx-ng'; Version='9.3.0' },
    @{ Name='libgomp'; Version='9.3.0' },
    @{ Name='ld_impl_linux-64'; Version='2.35.1' }
)
$wheels = @(
    @{ Name='numpy'; Version='1.20.3' },
    @{ Name='pandas'; Version='1.3.5' },
    @{ Name='scikit-learn'; Version='0.24.2' },
    @{ Name='scipy'; Version='1.7.3' },
    @{ Name='matplotlib'; Version='3.3.4' },
    @{ Name='python-dateutil'; Version='2.8.2' },
    @{ Name='pytz'; Version='2021.3' },
    @{ Name='joblib'; Version='1.0.1' },
    @{ Name='threadpoolctl'; Version='2.1.0' },
    @{ Name='Pillow'; Version='8.4.0' },
    @{ Name='cycler'; Version='0.10.0' },
    @{ Name='kiwisolver'; Version='1.3.1' },
    @{ Name='pyparsing'; Version='2.4.7' },
    @{ Name='six'; Version='1.16.0' },
    @{ Name='typing-extensions'; Version='3.10.0.2' },
    @{ Name='future'; Version='1.0.0' },
    @{ Name='setuptools'; Version='59.8.0' }
)
$requests = @($condaPackages | ForEach-Object { $_ + @{ Kind='conda' } }) + @($wheels | ForEach-Object { $_ + @{ Kind='wheel' } })
$manifest = $requests | ForEach-Object -Parallel {
    $request = $_
    $destinationRoot = $using:legacyRoot
    if ($request.Kind -eq 'conda') {
        $metadata = Invoke-RestMethod -Uri "https://api.anaconda.org/package/anaconda/$($request.Name)" -TimeoutSec 90
        $matching = @($metadata.files | Where-Object {
            $_.version -eq $request.Version -and $_.basename -like 'linux-64/*.tar.bz2' -and
            ((-not $request.Build) -or $_.attrs.build -eq $request.Build)
        })
        $file = $matching | Sort-Object {$_.attrs.build_number},basename | Select-Object -Last 1
        if (-not $file -or -not $file.sha256) { throw "No SHA256-pinned conda package for $($request.Name)" }
        $filename = Split-Path $file.basename -Leaf
        $url = 'https:' + $file.download_url
        $sha = $file.sha256
        $expected = $file.size
        $depends = $file.attrs.depends
    } else {
        $metadata = Invoke-RestMethod -Uri "https://pypi.org/pypi/$($request.Name)/$($request.Version)/json" -TimeoutSec 60
        $file = $metadata.urls | Where-Object {
            $_.packagetype -eq 'bdist_wheel' -and
            ($_.filename -match '-(py2\.py3|py3)-none-any\.whl$' -or
             $_.filename -match '-cp37-cp37m-.*manylinux(1|2010|2014|_2_(5|12|17))_x86_64.*\.whl$')
        } | Sort-Object filename | Select-Object -First 1
        if (-not $file) { throw "No compatible wheel for $($request.Name)" }
        $filename = $file.filename
        $url = $file.url
        $sha = $file.digests.sha256
        $expected = $file.size
        $depends = $metadata.info.requires_dist
    }
    $destination = Join-Path $destinationRoot $filename
    $valid = (Test-Path -LiteralPath $destination) -and
             ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -eq $sha)
    if (-not $valid) {
        & curl.exe -f -L -sS --connect-timeout 20 --max-time 600 --retry 3 --output $destination $url
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $filename" }
    }
    $actual = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $sha -or (Get-Item -LiteralPath $destination).Length -ne $expected) { throw "Verification failed: $filename" }
    [PSCustomObject]@{ kind=$request.Kind; package=$request.Name; version=$request.Version;
        filename=$filename; url=$url; sha256=$actual; bytes=$expected; dependencies=$depends }
} -ThrottleLimit 4
if (@($manifest).Count -ne $requests.Count) { throw 'Incomplete legacy environment assets' }
@($manifest | Sort-Object kind,package) | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $legacyRoot 'legacy_asset_manifest.json') -Encoding utf8
Write-Output "Verified $(@($manifest).Count) legacy runtime assets; CUDA torch wheel remains a separate official-source step."
