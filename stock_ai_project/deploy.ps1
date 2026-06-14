# Invest AI — Streamlit Cloud Deploy Script
# Run this AFTER: gh auth login

$ErrorActionPreference = "Stop"
$env:Path = "C:\Program Files\Git\bin;C:\Program Files\GitHub CLI;" + $env:Path

Set-Location $PSScriptRoot

Write-Host "=== Invest AI Deployment ===" -ForegroundColor Cyan

# Check GitHub login
gh auth status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Please log in to GitHub first:" -ForegroundColor Yellow
    Write-Host "  gh auth login" -ForegroundColor White
    Write-Host ""
    Write-Host "Then run this script again." -ForegroundColor Yellow
    exit 1
}

# Create GitHub repo and push (skip if remote already exists)
$remote = git remote get-url origin 2>$null
if (-not $remote) {
    Write-Host "Creating public GitHub repo: invest-ai ..." -ForegroundColor Green
    gh repo create invest-ai --public --source=. --remote=origin --push --description "Invest AI - AI-Powered Stock Market Decision Support System"
} else {
    Write-Host "Pushing to existing remote ..." -ForegroundColor Green
    git push -u origin main
}

$username = gh api user --jq .login
$repoUrl = "https://github.com/$username/invest-ai"

Write-Host ""
Write-Host "=== GitHub repo ready ===" -ForegroundColor Green
Write-Host "  $repoUrl"
Write-Host ""
Write-Host "=== Deploy on Streamlit Cloud ===" -ForegroundColor Cyan
Write-Host "1. Open: https://share.streamlit.io"
Write-Host "2. Sign in with GitHub"
Write-Host "3. Click 'Create app'"
Write-Host "4. Set:"
Write-Host "     Repository : $username/invest-ai"
Write-Host "     Branch     : main"
Write-Host "     Main file  : app.py"
Write-Host "     App URL    : invest-ai"
Write-Host "5. Click Deploy"
Write-Host "6. Settings -> General -> App name: Invest AI"
Write-Host "7. Settings -> Secrets -> add NEWS_API_KEY"
Write-Host ""
Write-Host "Your live URL will be:" -ForegroundColor Green
Write-Host "  https://invest-ai-$username.streamlit.app"
Write-Host ""
