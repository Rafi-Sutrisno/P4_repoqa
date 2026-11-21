# Dry-run Fase 3: retensi needle dan rasio token

Tanggal 2026-11-21, kode di commit `f868528`. Tanpa LLM. Cakupan: 600 tugas, jendela 16.384 token, token dihitung dengan tokenizer CodeLlama. Tahap `select` S2/S3 memakai `FakeClient`:

- `select_first`: selalu memilih ID 1..20. Ini pembanding naif, bukan perkiraan kinerja LLM.
- `oracle`: memilih unit yang signature-nya ada di needle. Ini perkiraan batas atas tahap seleksi.

Strategi tanpa LLM tidak dipengaruhi mode palsu. Hasil mentah ada di `results/ctx_dry_phase3/` (tidak di-commit).

```bash
python -m repoqa_ctx.run --strategy <s> --budget <B> [--strategy-args JSON] --dry-run [--fake-mode select_first|oracle]
python -m repoqa_ctx.summarize results/ctx_dry_phase3 --by-language
```

## Ringkasan (600 tugas)

Token masuk = semua panggilan per tugas: indeks `select` (S2/S3) ditambah prompt jawaban (instruksi + deskripsi + konteks). Sebagai pembanding, `full` = 15.370 token masuk dengan retensi 100% (Fase 2).

| Strategi | B | Token konteks | Rasio konteks | Token masuk/tugas | Panggilan | Retensi |
|---|---|---|---|---|---|---|
| rag_fn_bm25 | 1K / 2K / 4K | 1.009 / 2.026 / 4.040 | 15,0× / 7,5× / 3,7× | 1.245 / 2.262 / 4.276 | 1 | 55,8% / 64,2% / 73,7% |
| rag_fn_bm25_plus | 1K / 2K / 4K | 993 / 1.975 / 3.872 | 15,2× / 7,7× / 3,9× | 1.229 / 2.211 / 4.108 | 1 | **83,7% / 89,8% / 94,0%** |
| struct_bm25 (S1) | 1K / 2K / 4K | 993 / 1.974 / 3.871 | 15,2× / 7,7× / 3,9× | 1.229 / 2.210 / 4.107 | 1 | 72,3% / 82,0% / 89,5% |
| struct_bm25 skeleton | 1K / 2K / 4K | 1.003 / 1.991 / 3.892 | 15,1× / 7,6× / 3,9× | 1.239 / 2.227 / 4.128 | 1 | 72,3% / 82,0% / 89,5% |
| index_select sig (S2), oracle | 1K / 2K / 4K | 237 / 282 / 339 | 63,8× / 53,7× / 44,7× | 3.196 / 3.240 / 3.297 | 2 | 94,2% / 94,5% / 95,3% |
| index_select sig (S2), select_first | 1K / 2K / 4K | 830 / 1.266 / 1.606 | 18,2× / 12,0× / 9,4× | 3.788 / 4.225 / 4.565 | 2 | 21,8% / 24,5% / 26,5% |
| index_select doc (S2), oracle | 1K / 2K / 4K | 237 / 282 / 339 | sama dengan sig | 3.574 / 3.618 / 3.675 | 2 | sama dengan sig |
| cascade (S3), oracle | 1K / 2K / 4K | 295 / 383 / 473 | 51,3× / 39,5× / 32,0× | 1.622 / 1.710 / 1.800 | 2 | 86,7% / 87,0% / 87,8% |
| cascade (S3), select_first | 1K / 2K / 4K | 880 / 1.502 / 2.087 | 17,2× / 10,1× / 7,3× | 2.207 / 2.829 / 3.414 | 2 | 33,2% / 42,5% / 46,7% |
| rag_sw_bm25 256 | 1K / 2K / 4K | 908 / 1.940 / 3.990 | 16,7× / 7,8× / 3,8× | 1.144 / 2.177 / 4.226 | 1 | 47,7% / 65,3% / 78,0% |
| rag_sw_bm25 512 | 1K / 2K / 4K | 780 / 1.816 / 3.869 | 19,4× / 8,3× / 3,9× | 1.016 / 2.052 / 4.106 | 1 | 44,3% / 64,3% / 80,5% |

Waktu membangun konteks: ≤ 0,1 detik per tugas untuk semua strategi.

## Cara membaca

- **Retensi** = needle masih utuh di konteks (whitespace diabaikan). Ini perkiraan batas atas skor, bukan skor. Plafon penilaian RepoQA di ambang 0,8 = 98,7% (DECISIONS F19).
- **S2/S3 dengan oracle** bukan hasil metode, tetapi batas atas jika LLM memilih dengan sempurna. Kinerja nyata ada di antara `select_first` dan `oracle`, dan baru diketahui di lab.
- **Biaya S2.** Indeks semua fungsi menghabiskan ±3.000 token per tugas, jauh lebih mahal dari prompt jawabannya. Pada B = 1K, total token masuk S2 (±3.200) lebih besar daripada `rag_fn_bm25_plus` (±1.230). S3 memotong biaya itu menjadi ±1.700, tetapi batas atasnya 87%, karena dibatasi recall 20 kandidat teratas S1.
- **S1 lebih rendah dari `rag_fn_bm25_plus`** di semua anggaran. Menggabungkan empat field dengan RRF berbobot sama tampaknya melemahkan sinyal dari isi fungsi. Angka ini tidak dipakai untuk menyetel apa pun. Mengubah S1 adalah keputusanmu, dan sebaiknya dilakukan di split dev.
- **Skeleton** hanya menambah 10–20 token dan tidak mengubah retensi, karena unit utuh diambil lebih dulu dan sisa anggarannya kecil. Karena itu skeleton tidak dimasukkan ke `run_all.sh`.
- **`index=doc`** menambah ±380 token per tugas. Dengan oracle hasilnya identik dengan `sig`, karena oracle hanya melihat signature. Manfaat baris doc hanya bisa diukur dengan LLM sungguhan.
- **`rag_fn_bm25`** sedikit berubah dari Fase 2 (54,0 / 63,7 / 72,5%) karena parser baru. Definisinya tidak berubah (DECISIONS P2).

## Retensi per bahasa

Baris skeleton dan `index=doc` tidak ditampilkan karena identik dengan varian dasarnya.

| strategi | B | semua | python | cpp | java | typescript | rust | go |
|---|---|---|---|---|---|---|---|---|
| rag_fn_bm25 | 1024 | 55.8% | 72.0% | 43.0% | 45.0% | 78.0% | 36.0% | 61.0% |
| rag_fn_bm25 | 2048 | 64.2% | 78.0% | 51.0% | 56.0% | 81.0% | 41.0% | 78.0% |
| rag_fn_bm25 | 4096 | 73.7% | 84.0% | 61.0% | 68.0% | 83.0% | 56.0% | 90.0% |
| rag_fn_bm25_plus | 1024 | 83.7% | 86.0% | 84.0% | 82.0% | 86.0% | 75.0% | 89.0% |
| rag_fn_bm25_plus | 2048 | 89.8% | 92.0% | 87.0% | 89.0% | 92.0% | 86.0% | 93.0% |
| rag_fn_bm25_plus | 4096 | 94.0% | 91.0% | 92.0% | 98.0% | 95.0% | 94.0% | 94.0% |
| struct_bm25 | 1024 | 72.3% | 77.0% | 66.0% | 72.0% | 75.0% | 72.0% | 72.0% |
| struct_bm25 | 2048 | 82.0% | 78.0% | 78.0% | 82.0% | 85.0% | 85.0% | 84.0% |
| struct_bm25 | 4096 | 89.5% | 89.0% | 84.0% | 92.0% | 89.0% | 94.0% | 89.0% |
| index_select (select_first) | 1024 | 21.8% | 24.0% | 19.0% | 16.0% | 35.0% | 21.0% | 16.0% |
| index_select (oracle) | 1024 | 94.2% | 98.0% | 82.0% | 96.0% | 93.0% | 98.0% | 98.0% |
| index_select (select_first) | 2048 | 24.5% | 21.0% | 20.0% | 17.0% | 40.0% | 24.0% | 25.0% |
| index_select (oracle) | 2048 | 94.5% | 98.0% | 82.0% | 97.0% | 94.0% | 98.0% | 98.0% |
| index_select (select_first) | 4096 | 26.5% | 24.0% | 23.0% | 20.0% | 41.0% | 25.0% | 26.0% |
| index_select (oracle) | 4096 | 95.3% | 98.0% | 85.0% | 98.0% | 95.0% | 98.0% | 98.0% |
| cascade (select_first) | 1024 | 33.2% | 30.0% | 35.0% | 34.0% | 40.0% | 35.0% | 25.0% |
| cascade (oracle) | 1024 | 86.7% | 92.0% | 74.0% | 86.0% | 89.0% | 86.0% | 93.0% |
| cascade (select_first) | 2048 | 42.5% | 41.0% | 37.0% | 44.0% | 53.0% | 44.0% | 36.0% |
| cascade (oracle) | 2048 | 87.0% | 92.0% | 74.0% | 87.0% | 90.0% | 86.0% | 93.0% |
| cascade (select_first) | 4096 | 46.7% | 47.0% | 44.0% | 47.0% | 54.0% | 49.0% | 39.0% |
| cascade (oracle) | 4096 | 87.8% | 92.0% | 77.0% | 88.0% | 91.0% | 86.0% | 93.0% |
| rag_sw_bm25 | 1024 | 47.7% | 46.0% | 41.0% | 57.0% | 57.0% | 39.0% | 46.0% |
| rag_sw_bm25 | 2048 | 65.3% | 66.0% | 54.0% | 73.0% | 80.0% | 58.0% | 61.0% |
| rag_sw_bm25 | 4096 | 78.0% | 78.0% | 67.0% | 84.0% | 91.0% | 71.0% | 77.0% |
| rag_sw_bm25 (window=512) | 1024 | 44.3% | 44.0% | 41.0% | 55.0% | 52.0% | 32.0% | 42.0% |
| rag_sw_bm25 (window=512) | 2048 | 64.3% | 69.0% | 52.0% | 67.0% | 77.0% | 54.0% | 67.0% |
| rag_sw_bm25 (window=512) | 4096 | 80.5% | 81.0% | 74.0% | 78.0% | 92.0% | 73.0% | 85.0% |

## Rasio konteks per bahasa

Rasio hampir sama di semua bahasa, karena anggaran dipakai penuh. Pengecualiannya S2/S3 dengan oracle, karena konteksnya hanya unit yang terpilih.

| strategi | B | semua | python | cpp | java | typescript | rust | go |
|---|---|---|---|---|---|---|---|---|
| rag_fn_bm25 | 1024 | 15.0x | 15.3x | 15.1x | 15.1x | 15.0x | 15.5x | 14.0x |
| rag_fn_bm25 | 2048 | 7.5x | 7.6x | 7.5x | 7.5x | 7.4x | 7.7x | 6.9x |
| rag_fn_bm25 | 4096 | 3.7x | 3.8x | 3.8x | 3.8x | 3.8x | 3.9x | 3.5x |
| rag_fn_bm25_plus | 1024 | 15.2x | 15.5x | 15.1x | 15.1x | 16.2x | 15.7x | 13.9x |
| rag_fn_bm25_plus | 2048 | 7.7x | 7.7x | 7.6x | 7.6x | 8.3x | 7.9x | 6.9x |
| rag_fn_bm25_plus | 4096 | 3.9x | 3.9x | 3.9x | 3.9x | 4.4x | 4.0x | 3.5x |
| struct_bm25 | 1024 | 15.2x | 15.5x | 15.1x | 15.1x | 16.3x | 15.6x | 14.0x |
| struct_bm25 | 2048 | 7.7x | 7.7x | 7.6x | 7.6x | 8.3x | 7.9x | 6.9x |
| struct_bm25 | 4096 | 3.9x | 3.9x | 3.9x | 3.9x | 4.4x | 4.0x | 3.5x |
| index_select (oracle) | 1024 | 63.8x | 61.5x | 45.6x | 89.5x | 66.4x | 84.3x | 55.3x |
| index_select (oracle) | 2048 | 53.7x | 61.5x | 29.7x | 78.8x | 55.9x | 79.6x | 53.2x |
| index_select (oracle) | 4096 | 44.7x | 61.5x | 20.0x | 70.1x | 44.8x | 79.6x | 53.2x |
| cascade (oracle) | 1024 | 51.3x | 52.0x | 37.8x | 62.3x | 58.2x | 58.1x | 47.9x |
| cascade (oracle) | 2048 | 39.5x | 43.3x | 24.5x | 47.8x | 44.1x | 48.3x | 42.1x |
| cascade (oracle) | 4096 | 32.0x | 37.1x | 16.6x | 42.2x | 33.9x | 46.0x | 39.2x |
| rag_sw_bm25 | 1024 | 16.7x | 17.1x | 16.8x | 16.8x | 16.6x | 17.3x | 15.5x |
| rag_sw_bm25 | 2048 | 7.8x | 8.0x | 7.9x | 7.9x | 7.8x | 8.1x | 7.2x |
| rag_sw_bm25 | 4096 | 3.8x | 3.9x | 3.9x | 3.8x | 3.8x | 3.9x | 3.5x |
| rag_sw_bm25 (window=512) | 1024 | 19.4x | 19.9x | 19.5x | 19.5x | 19.3x | 20.1x | 18.1x |
| rag_sw_bm25 (window=512) | 2048 | 8.3x | 8.5x | 8.4x | 8.4x | 8.3x | 8.7x | 7.8x |
| rag_sw_bm25 (window=512) | 4096 | 3.9x | 4.0x | 4.0x | 3.9x | 3.9x | 4.0x | 3.6x |

## Perbandingan dengan pilot di PLAN.md

Pilot Fase 1 (PLAN §1.6) mengukur peringkat BM25 tanpa anggaran. Kandidatnya hanya fungsi dari `FUNCTION_QUERY` (rata-rata 46 per jendela), dengan stopword yang masih memuat label templat deskripsi.

| Ukuran | Pilot Fase 1 | Fase 3 (kandidat parser umum, 63 fungsi per jendela) |
|---|---|---|
| BM25 teks penuh, ±5 fungsi | recall@5 88% | `rag_fn_bm25_plus` B = 1K: retensi 83,7% |
| BM25 teks penuh, ±10 fungsi | recall@10 94% | `rag_fn_bm25_plus` B = 2K: retensi 89,8% |
| BM25 teks penuh, ±20 fungsi | recall@20 97% | `rag_fn_bm25_plus` B = 4K: 94,0%; batas S3 (top-20 S1): ±87% |
| Hanya signature | recall@5 79%, @10 87% | tidak diukur terpisah. Oracle S2 (indeks signature) 94%, tetapi itu batas atas seleksi, bukan BM25 |

Angka Fase 3 sedikit di bawah pilot. Ada tiga sebab yang diketahui:

- kandidatnya lebih banyak (parser umum menemukan 63 fungsi per jendela, bukan 46);
- anggaran B ikut menghitung header dan penanda `...`;
- stop set yang dipakai standar, tanpa label templat. Daftar label itu memakai pengetahuan tentang format deskripsi benchmark, jadi memang tidak boleh dipakai.

Urutannya tetap sama: BM25 pada isi fungsi dengan stopword adalah sinyal murah yang terkuat.
