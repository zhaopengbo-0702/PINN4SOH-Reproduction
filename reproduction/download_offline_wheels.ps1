$ErrorActionPreference = 'Stop'
$wheelRoot = Join-Path $PSScriptRoot 'wheels'
New-Item -ItemType Directory -Force -Path $wheelRoot | Out-Null
$packages = @(
    @{ Name = 'numpy'; Version = '1.24.4' },
    @{ Name = 'scipy'; Version = '1.10.1' },
    @{ Name = 'pandas'; Version = '1.5.3' },
    @{ Name = 'scikit-learn'; Version = '1.2.2' },
    @{ Name = 'matplotlib'; Version = '3.7.5' },
    @{ Name = 'SciencePlots'; Version = '2.1.1' },
    @{ Name = 'openpyxl'; Version = '3.1.5' },
    @{ Name = 'python-dateutil'; Version = '2.9.0.post0' },
    @{ Name = 'pytz'; Version = '2024.2' },
    @{ Name = 'joblib'; Version = '1.4.2' },
    @{ Name = 'threadpoolctl'; Version = '3.5.0' },
    @{ Name = 'contourpy'; Version = '1.2.1' },
    @{ Name = 'cycler'; Version = '0.12.1' },
    @{ Name = 'fonttools'; Version = '4.55.0' },
    @{ Name = 'kiwisolver'; Version = '1.4.7' },
    @{ Name = 'packaging'; Version = '24.2' },
    @{ Name = 'Pillow'; Version = '9.5.0' },
    @{ Name = 'pyparsing'; Version = '3.1.4' },
    @{ Name = 'et-xmlfile'; Version = '2.0.0' },
    @{ Name = 'six'; Version = '1.16.0' },
    @{ Name = 'importlib-resources'; Version = '6.4.5' },
    @{ Name = 'zipp'; Version = '3.20.2' },
    @{ Name = 'tzdata'; Version = '2024.2' }
)

$manifest = $packages | ForEach-Object -Parallel {
    $package = $_
    $root = $using:wheelRoot
    $jsonText = & curl.exe -f -sS --connect-timeout 15 --max-time 45 --retry 2 "https://pypi.org/pypi/$($package.Name)/$($package.Version)/json"
    if ($LASTEXITCODE -ne 0) { throw "Metadata download failed: $($package.Name)" }
    $metadata = $jsonText | ConvertFrom-Json
    $wheels = @($metadata.urls | Where-Object {
        $_.packagetype -eq 'bdist_wheel' -and (
            $_.filename -match '-(py2\.py3|py3)-none-any\.whl$' -or
            $_.filename -match '-cp310-cp310-.*manylinux(1|2010|2014|_2_(5|12|17))_x86_64.*\.whl$'
        )
    })
    if ($wheels.Count -eq 0) { throw "No compatible wheel for $($package.Name)" }
    $wheel = $wheels | Sort-Object filename | Select-Object -First 1
    $destination = Join-Path $root $wheel.filename
    $alreadyValid = (Test-Path -LiteralPath $destination) -and ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -eq $wheel.digests.sha256)
    if (-not $alreadyValid) {
        & curl.exe -f -L -sS --connect-timeout 15 --max-time 180 --retry 3 --output $destination $wheel.url
        if ($LASTEXITCODE -ne 0) { throw "Wheel download failed: $($wheel.filename)" }
    }
    $actual = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $wheel.digests.sha256) { throw "SHA256 mismatch: $($wheel.filename)" }
    [PSCustomObject]@{
        package = $package.Name
        version = $package.Version
        filename = $wheel.filename
        url = $wheel.url
        bytes = (Get-Item -LiteralPath $destination).Length
        sha256 = $actual
    }
} -ThrottleLimit 4

if (@($manifest).Count -ne $packages.Count) { throw 'Incomplete wheel set' }
$manifest = @($manifest | Sort-Object package)
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'offline-wheels-manifest.json') -Encoding utf8
$manifest | Select-Object package, version, bytes, filename | Format-Table -AutoSize
