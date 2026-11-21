# Pemilihan bobot struct_bm25 (split dev saja)

Split dev: 60 tugas (1 repo per bahasa, `repoqa_ctx/split.json`). Tanpa LLM; ukuran = retensi needle. Kandidat ditetapkan sebelum melihat hasil; urutan bobot: nama, signature, doc, isi. Skor = rata-rata retensi pada B 1024/2048/4096; seri dimenangkan kandidat yang lebih awal.

| bobot | B=1024 | B=2048 | B=4096 | rata-rata |
|---|---|---|---|---|
| [1, 1, 1, 1] | 70.0% | 75.0% | 81.7% | 75.6% |
| [2, 1, 1, 2] | 70.0% | 78.3% | 88.3% | 78.9% |
| [3, 1, 1, 3] **(terpilih)** | 71.7% | 78.3% | 88.3% | 79.4% |
| [2, 1, 1, 3] | 71.7% | 78.3% | 86.7% | 78.9% |

Terpilih: `{"weights": [3, 1, 1, 3]}`.
