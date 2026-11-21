# Fase 4: semua baseline, apa yang sudah terbukti jalan dan apa yang menunggu lab

Tanggal 2026-11-21, branch `p4-phase4`. Laptop tanpa GPU dan tanpa LLM. Semua angka di bawah berasal dari dry-run (tanpa model generator) atau uji asap di CPU.

## Sudah terbukti jalan di laptop

| Bagian | Bukti |
|---|---|
| Tes di `.venv` utama | 25 lulus, 1 dilewati (`test_compress_smoke` otomatis dilewati karena tidak ada torch) |
| Tes asap kompresi di `.venv-compress` (CPU, model kecil, 3 tugas per strategi) | **6/6 lulus** dalam 22,5 menit. Setiap strategi jalan, konteksnya ≤ B = 2048, dan hasilnya terbaca oleh runner (dry-run). |
| `.venv` utama membaca cache kompresi | dry-run 3 tugas × 6 strategi terkompresi berhasil di `.venv` (tanpa torch), semua ≤ 2048 token |
| Hitungan token antar-venv | CodeLlama identik di `.venv` (transformers 4.57.6) dan `.venv-compress` (4.44.2): 72 teks, hash sama |
| LongCodeZip resmi | Render kita identik dengan `compressed_code` resmi (`matches_official_render = True` di 3/3 tugas). Untuk masuk ≤ B token CodeLlama, 4–18 chunk berperingkat terendah dibuang (DECISIONS Q3). |
| Data resmi LongCodeZip | Ke-600 `code_context` identik dengan jendela `full` kita. Output Qwen2.5-Coder-7B bawaan repo resmi, dinilai dengan `compute_score` asli, = 86,0 dan sama per bahasa dengan Tabel IV paper, tetapi **hanya dengan `ignore_comments=True`** (DECISIONS T12, Q9). |
| `run_all.sh` | Rencana perintahnya dicetak lewat stub tanpa menjalankan apa pun: 86 run (43 per putaran) dan 24 tahap kompresi, masing-masing tepat sebelum strategi terkompresinya. Urutan tingkat A → B → C (Qwen) → A (DeepSeek), `--limit` 30 / 120 / penuh sesuai tingkat. |

### Biaya kompresi di CPU laptop (3 tugas, B = 2048; hanya indikasi, di GPU lab akan jauh lebih cepat)

| Strategi | Model kompresor | Detik/tugas | Token diproses kompresor/tugas | Catatan |
|---|---|---|---|---|
| rag_fn_unixcoder | unixcoder-base | 1–5 | ±8,6K | |
| rag_sw_unixcoder | unixcoder-base | 9–11 | ±28,6K | jendela 20/10 baris |
| longcodezip | Qwen2.5-Coder-0.5B | 36–45 | ±22,9K | 4–18 chunk dibuang agar ≤ B |
| llmlingua | Qwen2.5-Coder-0.5B | 92–194 | ±11,9K (perkiraan) | 1–2 kali coba target |
| longllmlingua | Qwen2.5-Coder-0.5B | 186–247 | ±11,9K (perkiraan) | 2 kali coba target |
| llmlingua2 | bert-base (uji asap) | 21–25 | ±15,9K (perkiraan) | **hasil mentok ±2,97K token berapa pun targetnya** (`force_tokens=["\n"]` mempertahankan setiap baris baru); ≤ B baru tercapai lewat pemotongan baris di akhir |

### Dry-run 600 tugas untuk strategi baru tanpa torch

| Strategi | B | Token konteks | Rasio | Retensi (600) | Retensi (test, 540) |
|---|---|---|---|---|---|
| struct_bm25 [3,1,1,3] | 1K / 2K / 4K | 992 / 1.973 / 3.868 | 15,3× / 7,7× / 3,9× | 74,0% / 83,0% / 91,8% | 74,3% / 83,5% / 92,2% |
| struct_bm25 [1,1,1,1] (Fase 3) | 1K / 2K / 4K | 993 / 1.974 / 3.871 | 15,2× / 7,7× / 3,9× | 72,3% / 82,0% / 89,5% | – |
| rag_fn_bm25_plus (Fase 3) | 1K / 2K / 4K | 993 / 1.975 / 3.872 | 15,2× / 7,7× / 3,9× | 83,7% / 89,8% / 94,0% | – |
| random_token | 1K / 2K / 4K | 1.001 / 2.004 / 4.005 | 15,1× / 7,6× / 3,8× | 0,0% / 0,0% / 1,2% | 0,0% / 0,0% / 1,3% |

Bobot [3,1,1,3] dipilih hanya di split dev (`reports/struct_weights_dev.md`). Di 600 tugas, bobot ini menaikkan retensi S1 sebesar 1–2 poin, tetapi S1 tetap di bawah `rag_fn_bm25_plus`.

Retensi strategi terkompresi belum diukur di 600 tugas. Di laptop, keluarga LLMLingua butuh 1,5–4 menit per tugas di CPU, sehingga 600 tugas terlalu lama. Pengukuran ini dilakukan di lab (tahap `compress` di `run_all.sh`). Retensi dari uji asap 3 tugas tidak bermakna secara statistik, jadi tidak dicantumkan.

## Baru bisa dibuktikan di lab

1. **Tahap kompresi di GPU dan kecepatannya**: `setup_compress.sh` (torch CUDA) dan `COMPRESS_DEVICE=auto`. GPU 12 GB dibagi antara Ollama dan kompresor (0,5B fp16 ±1 GB); belum diuji bersamaan.
2. **Retensi dan rasio di 600 tugas** untuk `rag_fn_unixcoder` dan `longcodezip`, serta di 120 tugas untuk `rag_sw_unixcoder` dan keluarga LLMLingua.
3. **Semua skor pass@1**: setiap strategi dengan Qwen2.5-Coder-7B dan DeepSeek-Coder-6.7B lewat Ollama, termasuk kinerja pilihan LLM di S2/S3 (sejauh ini hanya batas atas oracle).
4. **Model kompresor yang lebih besar** (misalnya Qwen2.5-Coder-1.5B lewat `--strategy-args '{"compressor_model": ...}'`) dan LLMLingua-2 xlm-roberta-large: belum pernah dijalankan.
5. **Perilaku Ollama**: `num_ctx`, offload DeepSeek, dan cache prefix (`scripts/lab_check.py`).

## Langkah di lab

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-ctx.txt
bash scripts/setup_compress.sh            # .venv-compress + LongCodeZip @617a30b
ollama pull qwen2.5-coder:7b-instruct && ollama pull deepseek-coder:6.7b-instruct
.venv/bin/python scripts/lab_check.py
bash scripts/run_all.sh 1                 # putaran cepat (--limit 30)
bash scripts/run_all.sh 2                 # putaran penuh
bash scripts/make_report.sh               # reports/<tanggal>_lab/
```
