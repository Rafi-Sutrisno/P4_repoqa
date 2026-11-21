# Keputusan dan TODO

Format: setiap keputusan berstatus `diusulkan` sampai kamu menyetujuinya. TODO berisi parameter yang tidak disebut sumbernya. Nilainya tidak boleh dikarang; isi hanya dari sumber yang bisa dirujuk, atau tandai sebagai "pilihan kita".

## Keputusan

| ID | Keputusan | Alasan | Status |
|---|---|---|---|
| D1 | Strategi hanya menerima jendela `code_context` dari `make_code_context` (default 16.384 token CodeLlama), bukan isi repo lain. | Semua metode memampatkan masukan yang sama dengan `full`, sehingga perbandingannya adil. | diusulkan |
| D2 | Tidak memperbaiki pergeseran offset byte/karakter di `make_code_context` (54/600 needle). Kode baru memakai `start_line`/`end_line`. | `full` harus identik dengan RepoQA asli. Isi konteks tidak terpengaruh. | diusulkan |
| D3 | Penilaian memakai `compute_score` asli dengan `ignore_comments=False` untuk semua strategi. Angka utama: ambang 0,8. | Penilaian bawaan tidak boleh berubah. Strategi yang membuang komentar harus menanggung akibatnya sendiri. | diusulkan |
| D4 | Hitungan token utama memakai tokenizer RepoQA (`codellama/CodeLlama-7b-Instruct-hf`). Hitungan backend (Ollama) disimpan sebagai pembanding. | Sebanding dengan `--code-context-size`, dan tidak terpengaruh cache prefix Ollama. | diusulkan |
| D5 | Kandidat unit diambil dengan parser tree-sitter yang lebih umum, bukan `FUNCTION_QUERY`, dan tanpa filter panjang atau keunikan nama. | Menghindari pemanfaatan cara benchmark dibangun. | diusulkan |
| D6 | RAG dijalankan dalam dua varian: BM25 (permintaanmu, menyimpang dari paper) dan UniXcoder-base (replikasi paper). Keduanya dilaporkan terpisah. | Paper memakai UniXcoder. BM25 bisa diuji di laptop. | diusulkan |
| D7 | `code_context_size` default 16.384. | Default RepoQA. Lihat hipotesis T13. | diusulkan |
| D8 | Tidak mengubah file asli. Kode baru masuk paket `repoqa_ctx/`, loop pembangunan tugas ditulis ulang, dan `test_full_identity` menjadi pengaman. | Perubahan minimal dan senyap di kode asli. | diusulkan |
| D9 | Panggilan LLM memakai API native Ollama dengan `num_ctx`, `temperature=0`, `seed`, dan `repeat_penalty=1` eksplisit, tanpa SIGALRM. | Jebakan #2, #6, dan #7 di PLAN.md §1.5. | diusulkan |
| D10 | DietCode dan SlimCode tidak direplikasi. Angka Tabel IV LongCodeZip disalin dengan catatan setup berbeda. | Keduanya dirancang untuk fungsi pendek dan bahasanya terbatas. | diusulkan |
| D11 | Strategi menerima ID opak, bukan `make_task_id` (yang memuat nama needle). | Mencegah bocoran. | diusulkan |
| D12 | Prompt jawaban selalu memakai TEMPLATE RepoQA; strategi hanya mengganti `code_context`. Pengecualian: S8 (pointer), yang dilaporkan di jalur terpisah. | Instruksi tetap sama, jadi perbedaan skor hanya berasal dari konteks. | diusulkan |
| D13 | Tuning hanya di split dev (1 repo per bahasa); laporan utama di split test. | Paper juga memakai held-out set terpisah. | diusulkan |
| D14 | Anggaran pembanding B ∈ {1K, 2K, 4K} token CodeLlama untuk semua metode yang memakai anggaran. 2K berasal dari paper; 1K dan 4K pilihan kita. | Memungkinkan kurva skor terhadap token. | diusulkan |

## TODO: parameter yang tidak disebut paper LongCodeZip

| ID | Item | Catatan |
|---|---|---|
| T1 | RAG Sliding Window: ukuran jendela dan stride | Tidak disebut. Usulan sapuan {256, 512} token dengan tumpang tindih 50% adalah pilihan kita. Cek repo resmi LongCodeZip. |
| T2 | RAG (keduanya): top-k atau anggaran, urutan potongan di konteks (asli atau menurut peringkat), granularitas sisa kode non-fungsi | Tidak disebut. Rasio tercapai di paper 3,5–4,5×. |
| T3 | Random Line dan Random Token: target rasio, perlakuan header `Path:` | Rasio tercapai 3,5× dan 3,6×, tetapi targetnya tidak disebut. |
| T4 | LLMLingua: model kompresor dan `rate`/`target_token` | Tidak disebut. Default pustaka perlu dicek pada versi yang dipasang. |
| T5 | LLMLingua-2: varian model (xlm-roberta-large atau bert-base-multilingual) | Tidak disebut. |
| T6 | LongLLMLingua: granularitas `context` (fungsi atau file) dan parameter (`rank_method`, `dynamic_context_compression_ratio`, `condition_compare`, `reorder_context`, `context_budget`) | Tidak disebut. |
| T7 | LongCodeZip untuk RepoQA: definisi q (instruksi saja, deskripsi saja, atau keduanya), format placeholder, urutan fungsi terpilih, chunker untuk 6 bahasa | Paper hanya menyebut B = 2K, R_fine = 1,0, dan model kompresor = model generator. Cek repo resmi. |
| T8 | UniXcoder: checkpoint persis dan cara pooling embedding | Paper menyebut "UniXCoder-base" (kemungkinan `microsoft/unixcoder-base`; belum dikonfirmasi). |
| T9 | Protokol "10 repeated experiments" untuk uji Wilcoxon: apa yang diacak? | Tidak dijelaskan. |
| T10 | Rasio kompresi: rasio dari rata-rata atau rata-rata rasio per sampel? Diukur dengan tokenizer apa? | Pers. (7) tidak menjelaskan cara agregasinya. Ringkasan kita melaporkan keduanya. |
| T11 | Inkonsistensi paper: skor RepoQA LongCodeZip pada Claude-3.7-Sonnet ditulis 90,7 di teks tetapi 88,9 di Tabel V | Pakai angka tabel, dan beri catatan. |
| T12 | Apakah No Compression di paper memakai `--clean-ctx-comments` atau setelan lain dari RepoQA? | Tidak disebut. Diasumsikan default (`none`). |
| T13 | Hipotesis: "Avg. Context Len. 11.524,6" di paper berasal dari jendela default 16K CodeLlama yang diukur dengan tokenizer model generator. Estimasi kita dengan tokenizer Qwen ±12,4K. | Belum terkonfirmasi. Menentukan apakah setelan kita sama dengan paper. |

## TODO: perlu diverifikasi di lab

| ID | Item |
|---|---|
| L1 | Default `num_ctx` dan opsi sampling Ollama pada versi yang terpasang; cara mengatur `num_ctx` untuk endpoint OpenAI-compatible (Modelfile atau env server). |
| L2 | Apakah `prompt_eval_count` Ollama mengecil ketika cache prefix dipakai ulang. |
| L3 | Apakah ada nondeterminisme pada temperature 0 (ulang `full` pada 50 tugas). |
| L4 | Template chat dan SYSTEM bawaan pada Modelfile model yang dipakai. |
