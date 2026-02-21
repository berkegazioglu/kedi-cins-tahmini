#!/usr/bin/env pwsh
# Docker Compose ile projeyi başlat

Write-Host "🐱 Kedi Cinsi Tahmin Sistemi - Docker Başlatma" -ForegroundColor Cyan
Write-Host "=" * 60

# Docker'ın çalışıp çalışmadığını kontrol et
Write-Host "`n📋 Docker durumu kontrol ediliyor..." -ForegroundColor Yellow
$dockerStatus = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker çalışmıyor! Lütfen Docker Desktop'ı başlatın." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Docker çalışıyor" -ForegroundColor Green

# Model dosyalarını kontrol et
Write-Host "`n📋 Model dosyaları kontrol ediliyor..." -ForegroundColor Yellow
$requiredFiles = @(
    "yolo11n.pt",
    "optimal_ensemble_final.pth",
    "EfficientNetB0_best.pth",
    "MobileNetV3_best.pth",
    "cat_breed_info.json"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file (EKSIK)" -ForegroundColor Red
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "`n⚠️  UYARI: Bazı model dosyaları eksik!" -ForegroundColor Yellow
    Write-Host "Eksik dosyalar: $($missingFiles -join ', ')" -ForegroundColor Yellow
    Write-Host "`nDevam etmek istiyor musunuz? (E/H): " -NoNewline
    $response = Read-Host
    if ($response -ne "E" -and $response -ne "e") {
        Write-Host "İptal edildi." -ForegroundColor Red
        exit 1
    }
}

# Docker Compose ile build ve start
Write-Host "`n🔨 Docker image'ı build ediliyor..." -ForegroundColor Yellow
Write-Host "Bu işlem ilk seferde birkaç dakika sürebilir...`n"

docker-compose build backend

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Build başarısız oldu!" -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ Build tamamlandı" -ForegroundColor Green

# Container'ları başlat
Write-Host "`n🚀 Container'lar başlatılıyor..." -ForegroundColor Yellow
docker-compose up -d backend

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Container başlatılamadı!" -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ Container'lar başlatıldı" -ForegroundColor Green

# Servislerin hazır olmasını bekle
Write-Host "`n⏳ Servisler başlatılıyor..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Logları göster
Write-Host "`n📜 Backend logları:" -ForegroundColor Cyan
Write-Host "=" * 60
docker-compose logs --tail=50 backend

Write-Host "`n" + "=" * 60
Write-Host "🎉 Sistem başarıyla başlatıldı!" -ForegroundColor Green
Write-Host "=" * 60
Write-Host "`n📍 Erişim Adresleri:" -ForegroundColor Cyan
Write-Host "   Backend API:     http://localhost:8000" -ForegroundColor White
Write-Host "   API Docs:        http://localhost:8000/docs" -ForegroundColor White
Write-Host "   Health Check:    http://localhost:8000/health" -ForegroundColor White
Write-Host "`n📋 Yararlı Komutlar:" -ForegroundColor Cyan
Write-Host "   Logları görüntüle:    docker-compose logs -f backend" -ForegroundColor White
Write-Host "   Durdur:               docker-compose down" -ForegroundColor White
Write-Host "   Yeniden başlat:       docker-compose restart backend" -ForegroundColor White
Write-Host "   Container'a gir:      docker-compose exec backend bash" -ForegroundColor White
Write-Host "`n" + "=" * 60
