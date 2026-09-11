# package_windows.ps1
# Builds and packages Nexir for easy distribution on Windows devices.

$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "           Nexir Packaging System                 " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Step 1: Build the UI binary in Release mode
Write-Host "[1/5] Building Nexir UI in release mode..." -ForegroundColor Yellow
cargo build -p ui --release --locked

# Step 2: Create distribution directory
$distDir = Join-Path $PSScriptRoot "nexir_dist"
if (Test-Path $distDir) {
    Remove-Item -Recurse -Force $distDir
}
New-Item -ItemType Directory -Path $distDir | Out-Null
Write-Host "[2/5] Created output directory: $distDir" -ForegroundColor Green

# Step 3: Copy the executable
$binSource = Join-Path $PSScriptRoot "target\release\ui.exe"
$binDest = Join-Path $distDir "nexir.exe"
Copy-Item $binSource $binDest
Write-Host "[3/5] Bundled executable: nexir.exe" -ForegroundColor Green

# Step 4: Copy FFmpeg DLLs
Write-Host "[4/5] Copying FFmpeg shared libraries from C:\ffmpeg\bin..." -ForegroundColor Yellow
$ffmpegBin = "C:\ffmpeg\bin"
$ffmpegDlls = @(
    "avcodec-62.dll",
    "avdevice-62.dll",
    "avfilter-11.dll",
    "avformat-62.dll",
    "avutil-60.dll",
    "swresample-6.dll",
    "swscale-9.dll"
)

foreach ($dll in $ffmpegDlls) {
    $dllPath = Join-Path $ffmpegBin $dll
    if (Test-Path $dllPath) {
        Copy-Item $dllPath (Join-Path $distDir $dll)
        Write-Host "  -> Copied $dll" -ForegroundColor DarkGray
    } else {
        Write-Warning "FFmpeg DLL not found: $dllPath"
    }
}

# Create a friendly README.txt
$readmeContent = @"
==================================================
                 NEXIR VIDEO EDITOR
==================================================

Nexir is a GPU-accelerated non-linear video editor written in Rust.

HOW TO RUN:
1. Double-click `nexir.exe` to launch the application.

SYSTEM REQUIREMENTS:
- Windows 10/11 (64-bit)
- DirectX 12 compatible GPU
- (Optional) NVIDIA GPU with recent drivers for hardware CUDA acceleration.
  Nexir will automatically detect your NVIDIA GPU and use it for ultra-fast
  hardware decoding/encoding. If no NVIDIA GPU is found, Nexir falls back 
  cleanly to multi-threaded CPU processing.
"@

Set-Content -Path (Join-Path $distDir "README.txt") -Value $readmeContent

# Step 5: Archive everything into a single zip file
$zipFile = Join-Path $PSScriptRoot "nexir-x86_64-pc-windows-msvc.zip"
if (Test-Path $zipFile) {
    Remove-Item $zipFile
}

Write-Host "[5/5] Creating zip archive: nexir-x86_64-pc-windows-msvc.zip..." -ForegroundColor Yellow
Compress-Archive -Path (Join-Path $distDir "*") -DestinationPath $zipFile

# Clean up temp dist folder
Remove-Item -Recurse -Force $distDir

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "Success! Nexir packaged into nexir-x86_64-pc-windows-msvc.zip" -ForegroundColor Green
Write-Host "Send this zip to anyone on Windows to run Nexir!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
