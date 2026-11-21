# Dry-run Fase 2: rasio token dan retensi needle

Tanggal 2026-11-21, kode di commit `e8a915c` (identik dengan `76317d3`; yang berbeda hanya dokumen). Tanpa LLM. Cakupan: 600 tugas RepoQA SNF (6 bahasa × 10 repo × 10 needle), jendela 16.384 token, token dihitung dengan tokenizer CodeLlama RepoQA.

```bash
python -m repoqa_ctx.run --strategy full --dry-run
python -m repoqa_ctx.run --strategy no_context --dry-run
python -m repoqa_ctx.run --strategy random_line --budget {1024,2048,4096} --dry-run
python -m repoqa_ctx.run --strategy rag_fn_bm25 --budget {1024,2048,4096} --dry-run
python -m repoqa_ctx.summarize results/ctx [--split test]
```

## Hasil

| Strategi | B | Token konteks (rata-rata) | Rasio vs full | Token masuk per tugas | Retensi needle (600) | Retensi needle (test, 540) |
|---|---|---|---|---|---|---|
| full | – | 15.134 | 1,0× | 15.370 | 100,0% | 100,0% |
| no_context | – | 0 | – | 237 | 0,0% | 0,0% |
| random_line | 1024 | 991 | 15,3× | 1.227 | 0,0% | 0,0% |
| random_line | 2048 | 1.983 | 7,6× | 2.219 | 0,7% | 0,6% |
| random_line | 4096 | 3.958 | 3,8× | 4.194 | 2,3% | 2,4% |
| rag_fn_bm25 | 1024 | 1.006 | 15,0× | 1.242 | 54,0% | 53,1% |
| rag_fn_bm25 | 2048 | 2.017 | 7,5× | 2.253 | 63,7% | 63,1% |
| rag_fn_bm25 | 4096 | 4.017 | 3,8× | 4.253 | 72,5% | 72,8% |

Waktu membangun konteks: ≤ 0,04 detik per tugas untuk semua strategi.

## Cara membaca

- **Retensi needle**: proporsi tugas yang masih memuat teks needle secara utuh di konteks (whitespace diabaikan; DECISIONS F15). Ini perkiraan kasar batas atas skor, bukan batas yang ketat. Needle yang kehilangan beberapa baris masih bisa lolos BLEU 0,8, dan needle yang utuh belum tentu disalin dengan benar. Skor sebenarnya baru diketahui setelah run di lab.
- **Rasio** dihitung atas token konteks saja (Σ jendela asli / Σ konteks; F16). **Token masuk** sudah termasuk instruksi (2×) dan deskripsi, ±236 token per tugas.
- Jendela `full` berisi 15.134 token bila teksnya ditokenisasi utuh. Anggaran 16.384 dihitung RepoQA per baris, dan setiap baris mendapat tambahan prefiks `▁`.
- **Batas atas penilaian.** Dengan jawaban sempurna, pass@1 di ambang 0,8 paling tinggi 592/600 (98,7%), karena BLEU RepoQA tidak bisa meloloskan beberapa needle yang sangat pendek (F19).
- **Cakupan parser.** 577/600 needle berada utuh dalam satu unit fungsi. Sisanya gagal di-parse karena jendela dimulai di tengah file (F9).

## Catatan per strategi

- `random_line` hampir tidak pernah menyisakan needle utuh, sehingga skor diperkirakan sangat rendah. Ini sejalan dengan paper LongCodeZip (7–12 pada 3,5×), walaupun angkanya tidak bisa dibandingkan langsung.
- `rag_fn_bm25` menyisakan needle pada 54–73% tugas. Diagnostik di bawah menunjukkan penyebabnya.

## Diagnostik `rag_fn_bm25` (tidak dipakai untuk tuning)

Peringkat needle di antara kandidat, dihitung pada 577 tugas yang needle-nya utuh dalam satu potongan. Diagnostik ini mencakup seluruh 600 tugas, termasuk test, sehingga **tidak boleh dijadikan dasar tuning**. Angka ini hanya menjelaskan penyebab, dan keputusan perubahannya ada di kamu.

| Kandidat dan kueri | recall@1 | @3 | @5 | @10 |
|---|---|---|---|---|
| fungsi + potongan kode, tanpa stopword (implementasi sekarang) | 23,4% | 39,6% | 50,4% | 64,9% |
| hanya fungsi, tanpa stopword | 36,6% | 59,7% | 69,3% | 84,0% |
| fungsi + potongan kode, dengan stopword | 50,3% | 72,7% | 80,3% | 87,0% |
| hanya fungsi, dengan stopword | 62,9% | 79,9% | 84,7% | 91,1% |

Dua faktor menurunkan retensi:

- Kata-kata umum dari deskripsi ("the", "of", "input", ...) cocok dengan potongan kode yang penuh komentar.
- Rata-rata 7,2 potongan non-fungsi berada di peringkat atas needle. Akibatnya, sebelum needle sempat terpilih, potongan-potongan itu sudah menghabiskan median 1.715 token.

Daftar stopword di diagnostik ini memuat label templat deskripsi (purpose, input, output, procedure). Daftar seperti itu memakai pengetahuan tentang format deskripsi benchmark, jadi tidak boleh dipakai apa adanya.
