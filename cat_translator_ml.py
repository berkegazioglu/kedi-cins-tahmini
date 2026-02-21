"""
Kedi Sesi Tercümanı - ML Pipeline
===================================
Kedi WAV dosyalarından özellik çıkarır, KMeans ile kümeler,
kümeler → duygusal kategoriler → JSON eşlemesi oluşturur.

Çıktı: frontend-react/public/cat_sound_mapping.json
       frontend-react/public/cat-sounds/*.wav
"""

import os
import json
import shutil
import numpy as np
import librosa
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(BASE_DIR, "Kedi Tercümanı", "cats_dogs")
TRAIN_CAT_DIR = os.path.join(DATASET_DIR, "train", "cat")
TEST_CAT_DIR  = os.path.join(DATASET_DIR, "test", "cats")
PUBLIC_DIR    = os.path.join(BASE_DIR, "frontend-react", "public")
SOUNDS_DIR    = os.path.join(PUBLIC_DIR, "cat-sounds")
OUTPUT_JSON   = os.path.join(PUBLIC_DIR, "cat_sound_mapping.json")

N_CLUSTERS = 7   # duygusal kategori sayısı
SAMPLE_RATE = 22050


def extract_features(wav_path: str) -> dict | None:
    """Bir WAV dosyasından akustik özellikler çıkarır."""
    try:
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        if len(y) < 1000:
            return None

        # Temel özellikler
        duration      = librosa.get_duration(y=y, sr=sr)
        rms           = float(np.mean(librosa.feature.rms(y=y)))
        zcr           = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spec_rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        spec_bw       = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

        # MFCC (13 katsayı)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Temel frekans (pitch) — YIN ile (pyin'den daha hızlı ve kararlı)
        try:
            f0 = librosa.yin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )
            # YIN sıfır tahminleri = voiceless → filtrele
            f0_voiced = f0[(f0 > 0) & (f0 < 2000)]
            mean_pitch  = float(np.mean(f0_voiced))  if len(f0_voiced) > 0 else 0.0
            pitch_range = float(np.ptp(f0_voiced))   if len(f0_voiced) > 0 else 0.0
        except Exception:
            mean_pitch, pitch_range = 0.0, 0.0

        feature_vec = np.concatenate([
            [duration, rms, zcr, spec_centroid, spec_rolloff, spec_bw,
             mean_pitch, pitch_range],
            mfcc_mean
        ])

        return {
            "file": os.path.basename(wav_path),
            "duration": duration,
            "rms": rms,
            "zcr": zcr,
            "mean_pitch": mean_pitch,
            "pitch_range": pitch_range,
            "spec_centroid": spec_centroid,
            "feature_vec": feature_vec.tolist(),
        }
    except Exception as e:
        print(f"  [HATA] {os.path.basename(wav_path)}: {e}")
        return None


def assign_labels_to_clusters(cluster_centers: np.ndarray) -> dict[int, str]:
    """
    Küme merkezlerindeki akustik özelliklere bakarak her kümeye
    bir duygusal etiket atar (kural tabanlı).

    Feature index referansı:
        0: duration      1: rms          2: zcr
        3: spec_centroid 4: spec_rolloff  5: spec_bw
        6: mean_pitch    7: pitch_range   8-20: mfcc
    """
    n = len(cluster_centers)
    DURATION  = 0
    RMS       = 1
    ZCR       = 2
    PITCH     = 6
    PITCH_RNG = 7

    labels_pool = [
        "selamlama",   # kısa, yüksek pitch
        "aclik",       # uzun, ısrarcı
        "oyun",        # enerjik, hızlı
        "kizginlik",   # yüksek enerji + ZCR
        "mutluluk",    # yumuşak, düşük enerji
        "soru",        # yükselen pitch
        "uzuntu",      # uzun, alçak
    ]

    # Her özellik için puan: yüksek=1, orta=0, düşük=-1
    def rank(arr):
        order = np.argsort(arr)
        r = np.zeros(n)
        for i, idx in enumerate(order):
            r[idx] = i  # 0 = en düşük, n-1 = en yüksek
        return r

    dur_rank   = rank(cluster_centers[:, DURATION])
    rms_rank   = rank(cluster_centers[:, RMS])
    zcr_rank   = rank(cluster_centers[:, ZCR])
    pit_rank   = rank(cluster_centers[:, PITCH])
    prng_rank  = rank(cluster_centers[:, PITCH_RNG])

    # Skor matrisi: cluster x label
    scores = np.zeros((n, len(labels_pool)))
    for c in range(n):
        scores[c, 0] += pit_rank[c] * 2   + prng_rank[c]         # selamlama: yüksek pitch
        scores[c, 1] += dur_rank[c] * 2   + rms_rank[c]          # açlık: uzun + enerji
        scores[c, 2] += zcr_rank[c] * 2   + rms_rank[c]          # oyun: yüksek ZCR + enerji
        scores[c, 3] += rms_rank[c] * 2   + zcr_rank[c] * 1.5    # kızgınlık: max enerji+ZCR
        scores[c, 4] += (n - 1 - rms_rank[c]) * 2                # mutluluk: düşük enerji
        scores[c, 5] += prng_rank[c] * 2  + pit_rank[c]          # soru: yüksek pitch range
        scores[c, 6] += dur_rank[c]       + (n - 1 - pit_rank[c])# üzüntü: uzun + düşük pitch

    # Greedy atama: en yüksek skorlu (cluster, label) çifti
    assigned_labels: dict[int, str] = {}
    used_labels = set()
    used_clusters = set()

    flat = [(scores[c, l], c, l) for c in range(n) for l in range(len(labels_pool))]
    flat.sort(reverse=True)

    for _, c, l in flat:
        if c in used_clusters or l in used_labels:
            continue
        assigned_labels[c] = labels_pool[l]
        used_clusters.add(c)
        used_labels.add(l)
        if len(assigned_labels) == min(n, len(labels_pool)):
            break

    # Etiket verilemeyen kümeler (N_CLUSTERS > label sayısı durumu)
    fallback_idx = 0
    for c in range(n):
        if c not in assigned_labels:
            while labels_pool[fallback_idx] in used_labels:
                fallback_idx += 1
            assigned_labels[c] = labels_pool[fallback_idx]
            used_labels.add(labels_pool[fallback_idx])

    return assigned_labels


def main():
    print("=" * 58)
    print("  Kedi Sesi Tercümanı — ML Pipeline")
    print("=" * 58)

    # 1. WAV dosyalarını topla
    wav_sources = []
    for d in [TRAIN_CAT_DIR, TEST_CAT_DIR]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".wav"):
                    wav_sources.append(os.path.join(d, f))

    print(f"\n[1/5] {len(wav_sources)} kedi WAV dosyası bulundu.")

    # 2. Özellik çıkarma
    print("[2/5] Akustik özellikler çıkarılıyor...")
    records = []
    for i, path in enumerate(wav_sources, 1):
        feat = extract_features(path)
        if feat:
            feat["src_path"] = path
            records.append(feat)
        if i % 20 == 0 or i == len(wav_sources):
            print(f"      {i}/{len(wav_sources)} tamamlandı")

    print(f"      → {len(records)} dosya başarıyla işlendi")

    # 3. KMeans kümeleme
    print(f"[3/5] KMeans kümeleme ({N_CLUSTERS} küme)...")
    X = np.array([r["feature_vec"] for r in records])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20, max_iter=500)
    labels_arr = kmeans.fit_predict(X_scaled)

    # Kümelere dosya ata
    clusters: dict[int, list] = {i: [] for i in range(N_CLUSTERS)}
    for rec, cluster_id in zip(records, labels_arr):
        clusters[cluster_id].append(rec)

    # 4. Kümelere duygusal etiket ata
    print("[4/5] Kümelere duygusal etiket atanıyor...")
    # Orijinal ölçekteki cluster centers (ters dönüştür)
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_label_map = assign_labels_to_clusters(centers_original)

    for cid, label in sorted(cluster_label_map.items()):
        print(f"      Küme {cid} → {label:15s} ({len(clusters[cid])} ses)")

    # 5. WAV dosyalarını public/cat-sounds/ klasörüne kopyala
    print(f"[5/5] WAV dosyaları kopyalanıyor → {SOUNDS_DIR}")
    os.makedirs(SOUNDS_DIR, exist_ok=True)

    mapping: dict[str, list[str]] = {}
    for cid, label in cluster_label_map.items():
        file_names = []
        for rec in clusters[cid]:
            dst = os.path.join(SOUNDS_DIR, rec["file"])
            if not os.path.exists(dst):
                shutil.copy2(rec["src_path"], dst)
            file_names.append(rec["file"])
        mapping[label] = sorted(file_names)

    # Duygusal kategori → Türkçe arama anahtar kelimeleri
    intent_keywords: dict[str, list[str]] = {
        "selamlama": [
            "merhaba", "selam", "günaydın", "iyi günler", "hey", "hoş",
            "nasılsın", "naber", "slm", "hi", "hello", "n'aber"
        ],
        "aclik": [
            "aç", "yemek", "ye", "besle", "mama", "yiyecek", "karnım",
            "açım", "öğle", "akşam", "sabah", "öğün", "atıştır"
        ],
        "oyun": [
            "oyna", "oyun", "top", "koş", "zıpla", "gel", "yakalamaca",
            "eğlen", "dans", "hareketli", "enerjik", "atla", "hoplay"
        ],
        "kizginlik": [
            "hayır", "dur", "kötü", "olmaz", "bırak", "yapma", "kız",
            "sinir", "kızgın", "öfke", "kızıyorum", "istemiyorum", "neden"
        ],
        "mutluluk": [
            "seviyorum", "güzel", "iyi", "aferin", "tatlı", "sarıl",
            "mutlu", "harika", "sevimli", "şirin", "süper", "müthiş"
        ],
        "soru": [
            "ne", "nerede", "kim", "nasıl", "niçin", "kaç", "hangi",
            "mi", "mı", "mu", "mü", "var mı", "acaba", "?"
        ],
        "uzuntu": [
            "üzgün", "ağla", "yalnız", "özledim", "gitti", "kayıp",
            "yas", "bitti", "yok", "olmadı", "kaybettim", "hayıf"
        ],
    }

    # Tüm çıktıyı tek bir JSON'a yaz
    output = {
        "version": "1.0",
        "total_sounds": len(records),
        "n_clusters": N_CLUSTERS,
        "mapping": mapping,
        "intent_keywords": intent_keywords,
        "cluster_stats": {
            cluster_label_map[cid]: {
                "count": len(clusters[cid]),
                "avg_duration": float(np.mean([r["duration"] for r in clusters[cid]])),
                "avg_pitch": float(np.mean([r["mean_pitch"] for r in clusters[cid]])),
                "avg_rms": float(np.mean([r["rms"] for r in clusters[cid]])),
            }
            for cid in range(N_CLUSTERS)
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Tamamlandı!")
    print(f"   JSON  → {OUTPUT_JSON}")
    print(f"   Sesler→ {SOUNDS_DIR}  ({len(os.listdir(SOUNDS_DIR))} dosya)")
    print()
    print("  Kategori özeti:")
    for label, files in sorted(mapping.items()):
        print(f"    {label:15s} {len(files):3d} ses")


if __name__ == "__main__":
    main()
