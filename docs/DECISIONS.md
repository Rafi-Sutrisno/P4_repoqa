# Keputusan dan TODO

Status:
- `disetujui`: kamu setujui secara eksplisit (pesan atau CLAUDE.md).
- `dipakai`: sudah diimplementasikan di Fase 2 sebagai opsi paling sederhana, menunggu konfirmasimu.
- `diusulkan`: belum diimplementasikan.

TODO berisi parameter yang tidak disebut oleh sumbernya. Nilainya tidak boleh dikarang; isi hanya dari sumber yang bisa dirujuk, atau tandai sebagai "pilihan kita".

## Keputusan Fase 1

| ID | Keputusan | Alasan | Status |
|---|---|---|---|
| D1 | Strategi hanya menerima jendela `code_context` dari `make_code_context` (16.384 token CodeLlama), bukan isi repo lain. | Semua metode memampatkan masukan yang sama dengan `full`, sehingga perbandingannya adil. | dipakai |
| D2 | Pergeseran offset byte/karakter di `make_code_context` (54/600 needle) tidak diperbaiki. Kode baru memakai `start_line`/`end_line`. | `full` harus identik dengan RepoQA asli. | dipakai |
| D3 | Penilaian memakai `compute_score` asli dengan `ignore_comments=False`. Angka utama di ambang 0,8; semua ambang disimpan. | Penilaian bawaan tidak boleh berubah. | disetujui |
| D4 | Token dihitung dengan tokenizer RepoQA (`codellama/CodeLlama-7b-Instruct-hf`). `prompt_eval_count` Ollama disimpan sebagai pembanding. | Sebanding dengan `--code-context-size` dan tidak terpengaruh cache prefix. | disetujui |
| D5 | Unit diambil dengan parser tree-sitter umum, bukan `FUNCTION_QUERY`, dan tanpa filter panjang atau keunikan nama. | Tidak memanfaatkan cara benchmark dibangun. | disetujui |
| D6 | RAG di fase ini hanya BM25. UniXcoder, LLMLingua, LongCodeZip, dan S4–S8 bukan bagian Fase 2. | Keputusanmu. | disetujui |
| D7 | `code_context_size` = 16.384. Model: `qwen2.5-coder:7b-instruct` dan `deepseek-coder:6.7b-instruct`. | Keputusanmu. | disetujui |
| D8 | File asli tidak diubah. Kode baru ada di `repoqa_ctx/`, loop pembangunan tugas ditulis ulang, dan `test_full_identity` menjadi pengaman. | Keputusanmu dan CLAUDE.md. | disetujui |
| D9 | Ollama dipanggil lewat API native `/api/chat` dengan opsi eksplisit, tanpa SIGALRM. Rincian di F14. | Jebakan #2, #6, #7 di PLAN.md §1.5. | dipakai |
| D10 | DietCode dan SlimCode tidak direplikasi; angka paper disalin. | Dirancang untuk fungsi pendek dengan bahasa terbatas. | diusulkan |
| D11 | Strategi menerima ID opak, bukan `make_task_id`. | CLAUDE.md. | disetujui |
| D12 | Prompt jawaban selalu memakai TEMPLATE RepoQA; strategi hanya mengganti `code_context`. | Perbedaan skor hanya berasal dari konteks. | dipakai |
| D13 | Tuning hanya di split dev; laporan di split test. Split dibuat dengan seed tetap dan di-commit (F7). | CLAUDE.md. | disetujui |
| D14 | Anggaran B ∈ {1K, 2K, 4K} token CodeLlama, dibaca sebagai 1024/2048/4096 (F12). | Keputusanmu. | disetujui |

## Keputusan Fase 2

| ID | Keputusan | Alasan | Status |
|---|---|---|---|
| F1 | Python 3.11.9 dipasang dengan winget (user scope). venv ada di `.venv/` (diabaikan git). | Laptop hanya punya 3.13; tree-sitter tidak punya wheel cp313. | dipakai |
| F2 | `transformers==4.57.6` dikunci (bukan 5.x). | Di v5, tokenizer CodeLlama tidak lagi menambahkan prefiks `▁` di awal teks. Pada sampel yang dicek jumlah tokennya sama, tetapi tokennya berbeda. RepoQA dibuat dengan 4.x. | dipakai |
| F3 | `repoqa_ctx.tasks.load_dataset` membaca dan menulis dataset sebagai UTF-8, dengan URL, versi, env override, dan path cache yang sama dengan `repoqa/data.py`. | `data.py` asli memakai encoding bawaan sistem; di Windows (cp1252) gagal karena isi dataset non-ASCII. Di Linux hasilnya identik. | dipakai |
| F4 | Cache tugas punya file sendiri, `CACHE_DIR/repoqa_ctx_tasks_ntoken_{size}_v1.jsonl`, dengan skema sama. | Dua pipeline tidak berbagi state. | dipakai |
| F5 | `uid = sha256(make_task_id(...))[:16]`. | ID opak yang deterministik. | dipakai |
| F6 | Selain uid, jendela, dan deskripsi, strategi juga menerima `language`, ditambah konfigurasi run (budget, params, seed). | `language` dibutuhkan untuk memilih parser tree-sitter. Ini bukan informasi needle karena sama untuk 100 tugas. **Ini menafsirkan aturan di CLAUDE.md, jadi butuh konfirmasimu.** | dipakai |
| F7 | Split dev = 1 repo per bahasa, `random.Random(0).choice` atas nama repo yang diurutkan, disimpan di `repoqa_ctx/split.json`. Setiap record memuat `split`; `summarize --split` default `all`. | Opsi paling sederhana dari PLAN §6. | dipakai |
| F8 | Unit fungsi: python `function_definition`; java `method_declaration`, `constructor_declaration`; ts `function_declaration`, `generator_function_declaration`, `method_definition` (plus `export`/`export default` di depannya); rust `function_item`; cpp `function_definition`; go `function_declaration`, `method_declaration`. Fungsi bersarang tetap di dalam unit induknya. Indentasi di awal baris ikut ke unit. Arrow function tidak dijadikan unit (tetap di potongan kode). | Parser umum (D5); arrow function dilewati demi kesederhanaan. | dipakai |
| F9 | Keterbatasan yang diketahui: 23/600 needle tidak tertangkap sebagai satu unit fungsi (16 berada di satu potongan kode, 7 terbelah). Penyebabnya parse gagal ketika jendela dimulai di tengah file, misalnya di dalam raw string Go atau pada indentasi Python. Tidak ditambal. | Opsi paling sederhana. Ini menjadi batas atas strategi berbasis unit fungsi. | dipakai |
| F10 | `random_line`: header `Path:` tidak pernah dibuang. Baris diacak dengan `Random(f"{seed}:{uid}")` dan dibuang sampai jumlah token per baris ≤ B (cara hitung RepoQA), lalu dicek ulang secara persis pada teks utuh. | Menyelesaikan T3 untuk implementasi kita. | dipakai |
| F11 | `rag_fn_bm25`: kandidat = unit fungsi + potongan kode tak kosong di antaranya. Kueri = field deskripsi apa adanya. BM25 Okapi k1 = 1,5, b = 0,75, idf = ln(1 + (N − df + 0,5)/(df + 0,5)) (**pilihan kita**, nilai umum, bukan dari paper). Token: kata alfanumerik dipecah camelCase/snake_case, huruf kecil, token 1 huruf dibuang, tanpa stopword. Greedy mengikuti peringkat dan melewati potongan yang tidak muat. Disusun dalam urutan asli dengan header file yang ikut dan `...\n` di celah. Header dan penanda ikut dihitung dalam B, dengan hitungan akhir yang persis. | Menyelesaikan T2 untuk implementasi kita. | dipakai |
| F12 | B = 1024, 2048, 4096. | 1K/2K/4K dibaca sebagai pangkat dua, seperti 16K = 16384. | dipakai |
| F13 | `num_ctx` = 24.576 untuk kedua model (1,5 × 16.384, angka yang sama dengan `max_model_len` vLLM di RepoQA). | Prompt `full` ±16,6K token CodeLlama + 1024 token keluaran. Risiko deepseek: lihat L5. | dipakai |
| F14 | Opsi Ollama: `temperature=0`, `seed=--seed` (0), `num_ctx`, `num_predict=1024`, `repeat_penalty=1.0`. Tanpa system message (sama dengan default RepoQA). Retry 3× untuk koneksi atau 5xx, timeout 900 detik. | D9. | dipakai |
| F15 | Retensi needle = teks acuan needle (per baris, sama dengan evaluator) ada utuh di konteks dengan **semua whitespace dibuang**. | `make_code_context` menyisipkan `\n` di awal needle. Pada 37 tugas dengan offset meleset, sisipan itu memotong token, sehingga perbandingan yang hanya menyeragamkan spasi menganggap needle hilang. Dengan definisi ini retensi `full` = 100%. | dipakai |
| F16 | Rasio = Σ token jendela asli / Σ token konteks (rasio dari jumlah). Keduanya dihitung dengan `count_tokens` atas teks utuh, bukan `code_context_ntokens`. | Menyelesaikan T10 untuk laporan kita; cara paper tetap tidak diketahui. | dipakai |
| F17 | Latensi per tugas = waktu bangun konteks + jumlah latensi semua panggilan LLM. Token masuk per tugas = jumlah token prompt semua panggilan (hitungan kita); saat dry-run, token prompt jawaban. | Definisi paling sederhana. | dipakai |
| F18 | Setiap run memakai folder bertimestamp baru. `--resume auto` melanjutkan run tidak selesai dengan identitas sama, atau melewatinya jika sudah selesai; `--resume DIR` melanjutkan folder itu. Tidak ada yang ditimpa: file konfigurasi dibuka dengan mode `x`, hasil hanya ditambah, baris terpotong dilewati, dan baris baru selalu mulai di baris baru. Dry-run selalu memakai klien palsu. `--limit` = N tugas pertama menurut urutan dataset. | Permintaanmu (langkah 5). | dipakai |
| F19 | `test_fake_pipeline` memeriksa pass@1 = 1,0 di ambang **0,0**, bukan 0,8. | BLEU RepoQA memberi 0,576 untuk salinan persis dari needle 3 token (`UTF8CharsetEncoder` di subset tes). Dengan jawaban sempurna, batas atas di ambang 0,8 adalah 592/600 (98,7%). Daftarnya: cpp `UTF8CharsetEncoder`, `uv_token`, `JsonConfigAdapter`, `HttpRouterTemplate`, `AuthorizationObject`; java `blankLines`, `enableForUri` (0,109); rust `__reject_custom_compilefail`. | dipakai |
| F20 | Rujukan penilaian: paper RepoQA membandingkan dengan semua fungsi F di konteks memakai BLEU > ambang, sedangkan kode membandingkan dengan 10 needle repo memakai ≥. Yang diikuti adalah kode (D3). | Penilaian bawaan tidak diubah. | dipakai |
| F21 | Commit dilakukan di branch `p4-phase2`, bukan `main`. Tidak di-push. | Aturan harness: jangan commit langsung di branch default. | dipakai |

## TODO: parameter yang tidak disebut paper LongCodeZip

| ID | Item | Catatan |
|---|---|---|
| T1 | RAG Sliding Window: ukuran jendela dan stride | Tidak disebut. Bukan bagian Fase 2. |
| T2 | RAG: top-k atau anggaran, urutan potongan, granularitas kode non-fungsi | Tidak disebut. Implementasi kita: F11 (pilihan kita). |
| T3 | Random Line dan Random Token: target rasio, perlakuan header | Tidak disebut. Implementasi kita: F10, memakai anggaran B. |
| T4 | LLMLingua: model kompresor dan `rate`/`target_token` | Tidak disebut. Bukan bagian Fase 2. |
| T5 | LLMLingua-2: varian model | Tidak disebut. Bukan bagian Fase 2. |
| T6 | LongLLMLingua: granularitas `context` dan parameter | Tidak disebut. Bukan bagian Fase 2. |
| T7 | LongCodeZip untuk RepoQA: definisi q, placeholder, urutan, chunker 6 bahasa | Paper hanya menyebut B = 2K, R_fine = 1,0, kompresor = generator. Bukan bagian Fase 2. |
| T8 | UniXcoder: checkpoint dan pooling | Bukan bagian Fase 2. |
| T9 | Protokol "10 repeated experiments" untuk Wilcoxon | Tidak dijelaskan. |
| T10 | Agregasi rasio di paper | Tidak dijelaskan. Laporan kita: F16. |
| T11 | Inkonsistensi paper: Claude-3.7-Sonnet 90,7 (teks) vs 88,9 (Tabel V) | Pakai angka tabel, dengan catatan. |
| T12 | Apakah No Compression di paper memakai `--clean-ctx-comments`? | Diasumsikan `none`. |
| T13 | Hipotesis: 11.524,6 token di paper = jendela 16K CodeLlama diukur dengan tokenizer generator (Qwen ±12,4K). | Belum terkonfirmasi. |

## TODO: perlu diverifikasi di lab (dicek oleh `scripts/lab_check.py`)

| ID | Item |
|---|---|
| L1 | Apakah `num_ctx` berlaku per request pada versi Ollama yang terpasang. |
| L2 | Apakah `prompt_eval_count` mengecil ketika prompt yang sama dikirim ulang (cache prefix). |
| L3 | Nondeterminisme pada temperature 0 (ulang `full` pada sebagian tugas). Belum ada skripnya. |
| L4 | Template chat dan SYSTEM bawaan Modelfile tiap model. |
| L5 | deepseek-coder-6.7b-instruct: konteks latihnya 16K, padahal prompt `full` + 1024 token keluaran bisa melewatinya. Selain itu ia memakai attention multi-head penuh, sehingga KV cache untuk 24K token kemungkinan tidak muat di 12 GB bersama bobot model, dan Ollama akan memindahkan sebagian ke CPU. `lab_check` melaporkan `fits_trained_context` dan porsi model di VRAM. Sebagai pembanding, paper RepoQA melaporkan 10,6% untuk model ini (dengan perluasan RoPE). |
