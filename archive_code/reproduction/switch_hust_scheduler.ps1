param([int]$ExistingPid = 24064, [ValidateSet('HUST','XJTU','Boundary')][string]$Worker='HUST')
$ErrorActionPreference = 'Stop'
$auditRoot = Join-Path $PSScriptRoot '../outputs/full_reproduction/feature_audit'
$auditRoot = [IO.Path]::GetFullPath($auditRoot)
$rawHust = if ($Worker -eq 'Boundary') { Join-Path $auditRoot 'boundary_extraction' } else { Join-Path $auditRoot ('raw_extraction/' + $Worker) }
$owned = Get-CimInstance Win32_Process -Filter "ProcessId = $ExistingPid"
$expectedCommand = if ($Worker -eq 'Boundary') { 'supplement_voltage_boundaries.py --watch' } else { 'extract_raw_feature_candidates.py --datasets ' + $Worker + ' --watch' }
if (-not $owned -or $owned.Name -ne 'python.exe' -or $owned.CommandLine -notmatch [Regex]::Escape($expectedCommand)) {
    throw 'PID is not the expected raw-data worker'
}
function Read-HustManifests {
    @(Get-ChildItem -LiteralPath $rawHust -Filter manifest.json -Recurse | ForEach-Object {
        $state = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
        [PSCustomObject]@{Path=$_.FullName; Status=$state.status; ProcessedCsv=$state.processed_csv}
    })
}
$before = Read-HustManifests
$baseline = @($before | Where-Object Status -eq 'complete').Count
$deadline = [DateTime]::UtcNow.AddMinutes(3)
while ([DateTime]::UtcNow -lt $deadline) {
    Start-Sleep -Milliseconds 100
    $states = Read-HustManifests
    $count = @($states | Where-Object Status -eq 'complete').Count
    if ($count -gt $baseline) {
        $current = Get-CimInstance Win32_Process -Filter "ProcessId = $ExistingPid"
        if (-not $current -or $current.CommandLine -ne $owned.CommandLine -or $current.CreationDate -ne $owned.CreationDate) { throw 'Worker identity changed' }
        Stop-Process -Id $ExistingPid
        Wait-Process -Id $ExistingPid -ErrorAction SilentlyContinue
        $after = Read-HustManifests
        $evidence = [ordered]@{
            stopped_at_utc=[DateTime]::UtcNow.ToString('o')
            stopped_pid=$ExistingPid
            stopped_process_created_at=$owned.CreationDate.ToUniversalTime().ToString('o')
            stopped_command_line=$owned.CommandLine
            complete_cells_before_boundary=$baseline
            complete_cells_at_boundary=@($after | Where-Object Status -eq 'complete').Count
            running_cells_to_recompute=@($after | Where-Object Status -eq 'running')
            calculation_implementation_changed=$false
            stopping_method='After observing natural completion, terminate the identified raw-data worker; any next-cell partial is recomputed exclusively'
        }
        $evidence | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $auditRoot ($Worker + '_scheduler_switch.json')) -Encoding utf8
        $evidence | ConvertTo-Json -Depth 8
        exit 0
    }
}
throw 'No completed cell boundary observed; original worker left running'
