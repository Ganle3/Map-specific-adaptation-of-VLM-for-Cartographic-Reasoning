param(
    [int]$MaxNewTokens = 500,
    [int]$LookupStartToken = 10,
    [int]$LookupEvery = 10,
    [int]$RelayLayer = 16,
    [int]$NumRelays = 2,
    [int]$LocalMaxPixels = 500000,
    [double]$InjectionScale = 0.10,
    [int]$Seed = 318
)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$vlmRoot = Join-Path $repoRoot "VLM_adaptation"
$python = Join-Path $env:USERPROFILE ".conda\envs\VisionGRPO\python.exe"
$adapter = Join-Path $vlmRoot "Training_outputs\MapVerse_scaling1000_GRPO_from_MapVerse5000_token500\checkpoint-2500"
$image = Join-Path $repoRoot "MapVerse\data\imgs\764.png"
$output = Join-Path $vlmRoot "Training_outputs\GRPO_relay\relay4_demo_qa318"

& $python (Join-Path $PSScriptRoot "demo_qa318.py") `
    --image $image --adapter $adapter --output-dir $output `
    --max-pixels 1000000 --local-max-pixels $LocalMaxPixels `
    --max-new-tokens $MaxNewTokens `
    --lookup-start-token $LookupStartToken --lookup-every $LookupEvery `
    --relay-layer $RelayLayer --num-relays $NumRelays `
    --injection-scale $InjectionScale --seed $Seed
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

