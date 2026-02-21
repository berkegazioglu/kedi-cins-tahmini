#!/usr/bin/env pwsh
# Docker Compose ile projeyi durdur ve temizle

Write-Host "🛑 Kedi Cinsi Tahmin Sistemi - Docker Durdurma" -ForegroundColor Cyan
Write-Host "=" * 60

# Container'ları durdur
Write-Host "`n📋 Container'lar durduruluyor..." -ForegroundColor Yellow
docker-compose down

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n⚠️  Container durdurulurken hata oluştu!" -ForegroundColor Yellow
} else {
    Write-Host "`n✅ Container'lar durduruldu" -ForegroundColor Green
}

# Temizlik seçeneği
Write-Host "`n🧹 Docker image'ını da silmek ister misiniz? (E/H): " -NoNewline
$response = Read-Host

if ($response -eq "E" -or $response -eq "e") {
    Write-Host "`n🗑️  Image siliniyor..." -ForegroundColor Yellow
    docker-compose down --rmi local
    Write-Host "✅ Image silindi" -ForegroundColor Green
}

Write-Host "`n" + "=" * 60
Write-Host "✅ Temizlik tamamlandı!" -ForegroundColor Green
Write-Host "=" * 60
