@echo off
setlocal EnableDelayedExpansion

REM This script downloads a specific dawn-build ZIP asset from:
REM   https://github.com/tjake/dawn-build/releases
REM
REM Usage:
REM   download_dawn.bat <BUILD_TAG> <PLATFORM> <ARCH> [TARGET_DIR]
REM
REM Parameters:
REM   TAG          - The release tag, e.g. "2025-03-23".
REM   PLATFORM     - One of: "linux", "mac", "windows".
REM   ARCH         - One of: "arm64", "x86_64", "x86" (where applicable).
REM   TARGET_DIR   - Directory to extract to (default: ".\dawn-native")
REM
REM Examples:
REM   REM Windows x86_64 debug build, extracts to .\dawn-native
REM   download_dawn.bat 2025-03-23 windows x86_64
REM
REM Requirements:
REM   - curl (included with Windows 10/11 or must be installed)
REM   - PowerShell (included with Windows)

REM You can adjust these if you fork or use a different repository.
set OWNER=tjake
set REPO=build-dawn

REM Read command line arguments
set TAG=%1
set PLATFORM=%2
set ARCH=%3
set TARGET_DIR=%4

REM Set default TARGET_DIR if not provided
if "%TARGET_DIR%"=="" set TARGET_DIR=.\dawn-native

REM Check mandatory parameters
if "%TAG%"=="" (
  echo Usage: %0 ^<TAG^> ^<PLATFORM^> ^<ARCH^> [TARGET_DIR]
  echo   BUILD_TAG: e.g. 2025-03-23
  echo   PLATFORM: linux, mac, windows
  echo   ARCH: x86_64, arm64, x86 (depends on platform)
  echo   TARGET_DIR: (optional) defaults to .\dawn-native
  exit /b 1
)
if "%PLATFORM%"=="" (
  echo Error: PLATFORM is required
  exit /b 1
)
if "%ARCH%"=="" (
  echo Error: ARCH is required
  exit /b 1
)

REM Construct the expected asset filename
set ASSET_NAME=dawn-%PLATFORM%-%ARCH%-%TAG%.zip
echo Downloading release asset '%ASSET_NAME%'...

REM Fetch release JSON via GitHub's API
set API_URL=https://api.github.com/repos/%OWNER%/%REPO%/releases/tags/%TAG%

REM Check for GITHUB_TOKEN environment variable
if "%GITHUB_TOKEN%"=="" (
  echo Warning: GITHUB_TOKEN environment variable not set. This may result in rate limiting.
  set CURL_CMD=curl -sS "%API_URL%"
) else (
  set CURL_CMD=curl -sS -H "Authorization: Bearer %GITHUB_TOKEN%" "%API_URL%"
)

REM Execute curl and save to temporary file
for /f "delims=" %%i in ('!CURL_CMD!') do set RELEASE_JSON=%%i

REM Extract the asset's download URL using PowerShell
set "PS_CMD=powershell -Command "$json = Get-Content -Raw | ConvertFrom-Json; $json.assets | Where-Object { $_.name -eq '%ASSET_NAME%' } | Select-Object -ExpandProperty browser_download_url""
for /f "delims=" %%u in ('!PS_CMD!') do set ASSET_URL=%%u

REM Check if the asset was found
if "!ASSET_URL!"=="" (
  echo Error: No matching asset found for '%ASSET_NAME%' in tag '%TAG%'.
  echo Check the spelling or ensure such an artifact exists in the release.
  exit /b 1
)

set TEMP_ZIP=%TEMP%\%ASSET_NAME%
echo Asset should be located (after download) at: %TEMP_ZIP%

REM Check if we already have the ZIP in TEMP
if exist "%TEMP_ZIP%" (
  echo File '%TEMP_ZIP%' already exists. Skipping download...
) else (
  echo File '%TEMP_ZIP%' not found. Downloading...
  curl -L -o "%TEMP_ZIP%" "!ASSET_URL!"
)

REM Create the target directory if it does not exist
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

echo Extracting '%ASSET_NAME%' into '%TARGET_DIR%'...
powershell -Command "Expand-Archive -Path '%TEMP_ZIP%' -DestinationPath '%TARGET_DIR%' -Force"

echo Download and extraction complete!
echo Files have been placed in: %TARGET_DIR%

endlocal
