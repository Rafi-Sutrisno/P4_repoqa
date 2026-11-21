# Rencana Fase 1: Manajemen Konteks untuk RepoQA Search Needle Function

Status: draf untuk disetujui. Belum ada kode yang diubah. Keputusan desain dan parameter yang belum pasti dicatat di [DECISIONS.md](DECISIONS.md).

## Ringkasan

- RepoQA SNF memberi model sebuah jendela kode sebesar 16.384 token (tokenizer CodeLlama) di sekitar satu fungsi target ("needle"), lalu meminta model menyalin fungsi itu. Tugas lulus jika keluaran paling mirip dengan needle yang benar dan BLEU-nya ≥ 0,8.
- Karena penilaian menuntut salinan hampir verbatim, metode yang memangkas isi fungsi (kompresi tingkat token seperti LLMLingua) hampir pasti gagal. Metode yang layak memilih fungsi utuh dan membuang sisanya.
- Dari statistik dataset: satu jendela 16K rata-rata memuat 46 fungsi (median 39) dengan panjang rata-rata ±197 token, sedangkan needle rata-rata 173 token. Sebagian besar isi jendela tidak dibutuhkan, asalkan fungsi yang benar bisa ditemukan.
- Pilot BM25 tanpa LLM (deskripsi dibandingkan dengan teks fungsi) menaruh needle di 5 besar pada 88% tugas dan di 10 besar pada 94%. Ini batas atas kasar untuk varian murah, bukan skor.
- Urutan usulan: (S1) BM25 berbasis struktur tanpa LLM tambahan, (S2) indeks signature → LLM memilih → baca kandidat (X1), (S3) kaskade BM25 → X1, lalu tambahan k adaptif, HyDE, verifikasi dua kandidat, tetangga graf, dan jawaban penunjuk.
- Semua kode baru masuk paket terpisah. Tidak ada baris kode asli RepoQA yang diubah. Kesetaraan strategi `full` dengan RepoQA asli dibuktikan lewat tes.

## 0. Catatan awal

- `CLAUDE.md` tidak ada di repo (sudah dicari di root dan seluruh subfolder). Rencana ini disusun tanpanya.
- `referensi/` hanya berisi `longcodezip.pdf`.
- Yang saya jalankan di laptop, seluruhnya di folder sementara di luar repo (tidak ada yang masuk git): mengunduh dataset RepoQA `2024-06-23` (12,5 MB), mengunduh file `tokenizer.json` CodeLlama dan Qwen2.5-Coder (hanya tokenizer, bukan bobot model), lalu menjalankan skrip statistik dan pilot BM25 tanpa LLM. Tidak ada model bahasa yang dijalankan.
- Statistik di bawah memakai tiruan logika jendela `make_code_context` yang saya tulis ulang tanpa dependensi RepoQA. Angkanya perkiraan dan bisa meleset beberapa token per tugas.

## 1. Cara kerja RepoQA

### 1.1 Dataset: lokasi unduhan, cache, dan bentuk item

Unduhan dan cache ada di [data.py](../repoqa/data.py):

- `_get_repoqa_data_ready_path()` ([data.py:19](../repoqa/data.py#L19)) mengunduh `https://github.com/evalplus/repoqa_release/releases/download/{VERSION}/repoqa-{VERSION}.json.gz` ([data.py:26](../repoqa/data.py#L26)), lalu menyimpannya sebagai JSON di `CACHE_DIR/repoqa-{VERSION}.json`.
- `CACHE_DIR = appdirs.user_cache_dir("repoqa")` ([data.py:13](../repoqa/data.py#L13)). Di Windows lokasinya `%LOCALAPPDATA%\repoqa\repoqa\Cache\`, di Linux `~/.cache/repoqa/`. Lokasi ini di luar repo, jadi cache tidak akan ikut ter-commit.
- Versi default `2024-06-23` ([data.py:16](../repoqa/data.py#L16)) bisa diganti lewat env `REPOQA_DATA_VERSION`. File lokal bisa dipakai lewat env `REPOQA_DATA_OVERRIDE_PATH` ([data.py:15](../repoqa/data.py#L15)) atau argumen `--dataset-path` ([search_needle_function.py:398](../repoqa/search_needle_function.py#L398)).
- Isinya 6 bahasa × 10 repo × 10 needle = 600 tugas. README masih menyebut 500 tugas dan 5 bahasa karena Go baru ditambahkan di commit `603cb73`.

Bentuk dataset (field diperiksa langsung dari file):

```text
{ "<bahasa>": [                       # python, cpp, java, typescript, rust, go
  { "repo", "commit_sha", "entrypoint_path", "topic",
    "content":    { "<path>": "<isi file>" },
    "dependency": { "<path>": ["<path yang diimpor>", ...] },
    "functions":  { "<path>": [ {name, start_line, end_line, start_byte, end_byte,
                                 global_start_line, global_end_line,
                                 global_start_byte, global_end_byte, code_ratio} ] },
    "needles":    [ {name, start_line, end_line, start_byte, end_byte, global_*,
                     code_ratio, path, description} ]          # 10 per repo
  } ] }
```

- `description` dibuat oleh GPT-4 ([needle_annotation.py:18](../scripts/curate/needle_annotation.py#L18)) dalam 4 bagian Purpose, Input, Output, Procedure. GPT-4 dilarang menyebut nama fungsi dan nama variabel. Contoh untuk `_merge_string_group` (psf/black): *"Purpose: To combine adjacent strings into a single string within a line of code..."*.
- Kriteria pemilihan needle ([needle_selection.py:56-71](../scripts/curate/needle_selection.py#L56-L71)): nama unik di repo, panjang ≤ 2000 byte, dan tiap needle berada di bin posisi yang berbeda.

Satu item tes dibangun di `evaluate_model` ([search_needle_function.py:497-515](../repoqa/search_needle_function.py#L497-L515)):

```text
{ "repo", "name", "language", "path",
  "position_ratio": (i + 0.5) / 10,            # needle ke-i diletakkan di 5%, 15%, ..., 95%
  "description": "\nFunction Description:<deskripsi>\n",
  "instruction": INSTRUCTION, "template": TEMPLATE,
  "code_context": "<jendela kode>",
  "needle_token_start", "needle_token_end", "code_context_ntokens" }
```

Item ini juga disimpan ke `CACHE_DIR/cache_ntoken_{size}_v1.jsonl` ([search_needle_function.py:432](../repoqa/search_needle_function.py#L432)) dengan kunci `lang::repo::name::size::ratio`.

### 1.2 Penyusunan konteks dan prompt (titik sisip `--strategy`)

1. Urutan file mengikuti `topological_sort(repo["dependency"])` ([utility.py:58](../repoqa/utility.py#L58)), yang dipanggil di [search_needle_function.py:473](../repoqa/search_needle_function.py#L473).
2. `make_code_context` ([search_needle_function.py:255](../repoqa/search_needle_function.py#L255)):
   - Anggaran prefix = `int(size × position_ratio − ntok_needle / 2)`, anggaran suffix = sisanya ([baris 287-288](../repoqa/search_needle_function.py#L287-L288)).
   - Prefix diisi mundur, baris demi baris, dari file needle ([baris 291](../repoqa/search_needle_function.py#L291)). Setelah itu file-file sebelumnya diisi maju dari awal file ([baris 300-312](../repoqa/search_needle_function.py#L300-L312)). Akibatnya, jika anggaran habis di tengah file sebelumnya, bagian akhir file itu hilang dan konteks tidak lagi bersambung.
   - Setiap file diberi header komentar `# Path: <path>` atau `// Path: <path>`. File yang terpotong diakhiri `...`.
   - Opsi `--clean-ctx-comments` (`none` / `positional_padding` / `no_padding`) menghapus komentar. Default-nya `none`.
3. Prompt dirakit di [baris 581-583](../repoqa/search_needle_function.py#L581-L583): kunci `TEMPLATE = "instruction\ncode_context\ndescription\ninstruction"` digabung tanpa pemisah. Hasilnya `INSTRUCTION + code_context + "\nFunction Description:...\n" + INSTRUCTION`. Instruksi muncul dua kali, dan tidak ada baris baru di antara instruksi pertama dan konteks.
4. Panggilan model: `engine.generate_reply(prompt, n=1, max_tokens=1024, system_msg=None)` ([baris 585](../repoqa/search_needle_function.py#L585)). Semua provider memakai temperature 0 secara default.

Titik sisip strategi ada setelah `task.update(code_context_info)` ([baris 515](../repoqa/search_needle_function.py#L515)) dan sebelum prompt dirakit ([baris 581](../repoqa/search_needle_function.py#L581)). Strategi menerima `code_context` asli dan mengembalikan `code_context` baru. Prompt jawaban tetap dirakit dengan `TEMPLATE` yang sama, sehingga instruksi dan deskripsi tidak berubah antarstrategi. Rencananya file asli tidak disentuh. Runner baru memakai ulang `make_code_context`, `TEMPLATE`, `INSTRUCTION`, dan `compute_score` dari modul asli, lalu menulis ulang loop pembangunan tugas yang hanya ±30 baris (§5.1).

### 1.3 Penghitungan token

- Tokenizer: `AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")` ([baris 271](../repoqa/search_needle_function.py#L271)), dimuat ulang setiap kali `make_code_context` dipanggil. README keliru menyebut tokenizer DeepSeekCoder.
- Cara hitung: `len(tokenizer.tokenize(line + "\n"))` per baris ([baris 46](../repoqa/search_needle_function.py#L46), [61](../repoqa/search_needle_function.py#L61)), tanpa token spesial. `code_context_ntokens` kemudian dihitung ulang dari teks utuh ([baris 353-355](../repoqa/search_needle_function.py#L353-L355)), jadi bisa sedikit berbeda dari jumlah per baris.
- `--code-context-size` (default 16384) adalah anggaran untuk `code_context` saja. Instruksi (29 token, muncul dua kali) dan deskripsi (rata-rata 179 token) berada di luar anggaran itu. Total prompt ≈ 16,6K token CodeLlama.
- Model sendiri melihat jumlah token menurut tokenizer-nya. Untuk Qwen2.5-Coder, rasio token Qwen/CodeLlama pada dataset ini ≈ 0,76, sehingga jendela 16K setara ±12,4K token Qwen.
- Backend vLLM memakai `max_model_len = 1,5 × size` ([baris 545](../repoqa/search_needle_function.py#L545)).

### 1.4 Penilaian dan format file hasil

`compute_score` ([compute_score.py:259](../repoqa/compute_score.py#L259)) memanggil `needle_evaluator` ([compute_score.py:154](../repoqa/compute_score.py#L154)) untuk setiap tugas:

1. `sanitize_output` ([compute_score.py:84](../repoqa/compute_score.py#L84)) mengambil blok kode berpagar ```` ``` ````, lalu memakai `FUNCTION_QUERY` tree-sitter untuk mengambil fungsi pertama di dalam blok. Jika tidak ada blok kode, keluaran mentah yang dipakai.
2. Keluaran dibandingkan dengan 10 needle repo itu saja, bukan dengan semua fungsi. Teks acuan diambil per baris: `content.split("\n")[start_line:end_line]`.
3. Kemiripan dihitung dengan BLEU nltk, token dipisah spasi, smoothing method4 ([metric.py:10](../repoqa/metric.py#L10)).
4. Tugas lulus jika needle termirip adalah needle target dan BLEU ≥ ambang. Ambang dihitung untuk 0,0 sampai 1,0 ([compute_score.py:27](../repoqa/compute_score.py#L27)). Angka utama di README dan paper LongCodeZip adalah 0,8. Dengan n=1, pass@1 = proporsi tugas yang lulus.
5. `ignore_comments` aktif otomatis jika `clean_ctx_comments ≠ none` ([search_needle_function.py:599](../repoqa/search_needle_function.py#L599)).

File hasil:

- `results/ntoken_{size}/{model}.jsonl` ([search_needle_function.py:406-411](../repoqa/search_needle_function.py#L406-L411), `/` pada nama model diganti `_slash_`). Satu baris per tugas berisi item tes ditambah `"output": [jawaban]`. File ini juga dipakai untuk melanjutkan run yang terputus.
- `{model}-SCORES.json`:

```text
{ "<model>": { "eval_date", "train_size",
    "scores":  { "all": { "0.8": {"pass@1": x}, ... }, "<bahasa>": {...} },
    "results": { "<bahasa>": [ {repo, name, needle_position, is_best_similar,
                                best_similar_score, best_target,
                                position: {token_start, token_end}} ] } } }
```

- Skor bisa dihitung ulang secara terpisah dengan `repoqa.compute_score --model-output-path <file>.jsonl`.

### 1.5 Jebakan yang ditemukan

| # | Masalah | Lokasi | Dampak | Penanganan |
|---|---|---|---|---|
| 1 | Offset byte dari tree-sitter dipakai sebagai indeks karakter Python | [search_needle_function.py:279](../repoqa/search_needle_function.py#L279) | Pada 54/600 needle (9%), potongan `needle_code` bergeser karena ada karakter non-ASCII; 1 needle bahkan kosong. Isi konteks tetap utuh (prefix + needle + suffix bersambung), yang meleset hanya posisi dan `needle_token_*`. | Tidak diperbaiki, karena `full` harus identik. Kode baru memakai `start_line`/`end_line`, sama seperti evaluator. |
| 2 | `signal.SIGALRM` di request provider | [request/openai.py:43](../repoqa/provider/request/openai.py#L43) (juga anthropic, google) | Windows tidak punya SIGALRM. `AttributeError` tertangkap oleh `except Exception`, sehingga loop berjalan tanpa akhir. | Wrapper LLM sendiri dengan timeout HTTP. |
| 3 | `save_json` memanggil `input()` jika file skor sudah ada | [compute_score.py:366](../repoqa/compute_score.py#L366) | Run batch bisa macet. | Skrip ringkasan memanggil `compute_score` langsung. |
| 4 | `fetch_hf_context` mengunduh config HF jika nama model mengandung `/` | [compute_score.py:221](../repoqa/compute_score.py#L221) | Nama seperti `hf.co/...` memicu unduhan dan bisa gagal. | Runner memakai nama tampilan tanpa `/`. |
| 5 | Nama model Ollama mengandung `:` | penamaan file [baris 410](../repoqa/search_needle_function.py#L410) | Karakter `:` tidak sah dalam nama file Windows. | Sanitasi nama file. |
| 6 | `num_ctx` default Ollama jauh di bawah 16K | (di luar repo) | Prompt bisa terpotong tanpa peringatan. | Set `num_ctx` eksplisit lewat API native per request. Endpoint OpenAI-compatible Ollama tidak menerima `num_ctx` per request, jadi perlu Modelfile atau env server (cek versi Ollama di lab). |
| 7 | Default sampling Ollama (mis. `repeat_penalty`) bisa berbeda dari greedy HF/vLLM | (di luar repo) | Penalti pengulangan bisa merusak salinan verbatim. | Set eksplisit `temperature=0`, `seed`, `repeat_penalty=1`, lalu catat semua opsi di log. Perlu diverifikasi di lab. |
| 8 | `tree_sitter<=0.21.3` dan `tree_sitter_languages 1.10.2` hanya punya wheel sampai cp312 | [setup.cfg](../setup.cfg) | Laptop memakai Python 3.13.7, jadi paket ini tidak bisa dipasang. | Pakai venv Python 3.11 atau 3.12 di laptop dan di lab. |
| 9 | `FUNCTION_QUERY` sempit: TS hanya `function_declaration`, Java hanya `method_declaration`, C++ hanya definisi dengan identifier polos | [utility.py:13](../repoqa/utility.py#L13) | Semua needle cocok dengan kueri ini. Strategi yang membatasi kandidat ke kueri ini memanfaatkan cara benchmark dibuat (bocoran). | Kandidat strategi diambil dengan parser yang lebih umum: method, konstruktor, arrow function, definisi C++ berkualifikasi. |
| 10 | Kriteria needle (nama unik, ≤ 2000 byte) juga pengetahuan tentang konstruksi benchmark | [needle_selection.py:56-71](../scripts/curate/needle_selection.py#L56-L71) | Sama dengan #9. | Strategi tidak boleh memfilter dengan kriteria ini. |
| 11 | `task_id` asli memuat nama needle | [search_needle_function.py:365](../repoqa/search_needle_function.py#L365) | Jika diteruskan ke strategi, nama needle bocor. | Strategi hanya menerima ID opak (§5.2). |

### 1.6 Statistik jendela 16K (perkiraan, 600 tugas)

"Fungsi" di tabel ini adalah entri `functions` dataset, yang dibuat dengan `FUNCTION_QUERY`. Untuk C++ dan TS, kueri ini hanya menangkap 17–22% token jendela, jadi recall pilot untuk dua bahasa itu terlalu optimis (kandidatnya lebih sedikit dari kenyataan).

| Bahasa | Fungsi per jendela (rata-rata / p90) | Token needle (rata-rata) | Porsi token fungsi di jendela | Total token signature | BM25 teks penuh @1 / @5 / @10 | BM25 signature saja @5 / @10 |
|---|---|---|---|---|---|---|
| python | 66,9 / 108 | 236 | 0,72 | 957 | 0,64 / 0,91 / 0,94 | 0,76 / 0,89 |
| cpp | 19,0 / 41 | 117 | 0,17 | 321 | 0,74 / 0,93 / 0,98 | 0,89 / 0,95 |
| java | 78,5 / 138 | 114 | 0,43 | 770 | 0,56 / 0,81 / 0,88 | 0,70 / 0,76 |
| typescript | 19,2 / 38 | 174 | 0,22 | 220 | 0,83 / 0,94 / 1,00 | 0,90 / 0,95 |
| rust | 63,6 / 97 | 162 | 0,53 | 1127 | 0,47 / 0,75 / 0,84 | 0,64 / 0,74 |
| go | 30,8 / 50 | 235 | 0,38 | 484 | 0,73 / 0,94 / 0,98 | 0,84 / 0,94 |
| semua | 46,3 / 93 | 173 (median 110, p90 432, maks 924) | 0,41 | 647 (p90 1354) | 0,66 / 0,88 / 0,94 (@20: 0,97) | 0,79 / 0,87 |

Catatan:

- Selain target, rata-rata ada 1 needle lain di jendela yang sama (p90: 3). Needle-needle inilah pesaing langsung dalam penilaian.
- Rata-rata ada 8,9 file per jendela.
- Pilot BM25: identifier dipecah (camelCase dan snake_case), stopword sederhana, k1 = 1,5, b = 0,75 (nilai umum, belum dituning). Ini pilot, bukan hasil eksperimen.

## 2. Ringkasan `referensi/`

Folder ini hanya berisi satu file: `longcodezip.pdf` (Shi dkk., arXiv 2510.00446, Okt 2025). Semua bagiannya relevan. Yang saya ambil:

| Bagian | Yang diambil |
|---|---|
| §III-A, Pers. (7) | Definisi rasio kompresi = token konteks asli / token konteks terkompresi. |
| §IV-C | Daftar dan definisi baseline, sama persis dengan daftarmu. RAG di paper memakai **UniXcoder-base** + cosine, bukan BM25. DietCode hanya untuk Python/Java; SlimCode hanya Java (paper mereproduksinya untuk Python). |
| §IV-B, Tabel I | RepoQA di paper: 600 tes, 6 bahasa, rata-rata konteks 11.524,6 token. Tokenizer yang dipakai tidak disebut. |
| §IV-D | Metrik RepoQA: BLEU > 0,8 (kode RepoQA memakai ≥; selisihnya tidak berarti). |
| §IV-E | Setelan RepoQA: anggaran B = 2K token, R_fine = 1,0 (tahap halus praktis mati, supaya fungsi tetap utuh). Model kompresor sama dengan model generator. Hiperparameter dituning di held-out set kecil yang tidak tumpang tindih dengan data uji. |
| §III-C | Di RepoQA, LongCodeZip pada dasarnya hanya tahap kasar: fungsi diperingkat dengan AMI = PPL(q) − PPL(q\|c), lalu dipilih greedy dalam 2K token, dan fungsi yang dibuang diganti placeholder. |
| Tabel VII (ablasi, code completion) | Pemeringkatan kasar paling berpengaruh: diganti similarity turun 7,89 ES, diganti acak turun 17,79. Ini mendukung fokus kita pada sinyal pemeringkatan fungsi. |
| Tabel VIII | Kompresor 0,5B tetap bekerja (diuji di code completion, bukan RepoQA). |
| Tabel IV, V | Angka acuan RepoQA (di bawah). |

Angka RepoQA dari paper (rata-rata 6 bahasa, rasio kompresi dalam kurung):

| Metode | DeepSeek-Coder-6.7B | Qwen2.5-Coder-7B | Seed-Coder-8B |
|---|---|---|---|
| No Compression | 38,3 | 86,0 | 69,0 |
| No Context | 0,0 | 0,0 | 0,0 |
| Random Token | 1,7 (3,6×) | 3,5 (3,6×) | 3,5 (3,6×) |
| Random Line | 7,3 (3,5×) | 11,8 (3,5×) | 9,3 (3,5×) |
| RAG Sliding Window | 55,5 (3,5×) | 67,5 (3,7×) | 61,7 (3,9×) |
| RAG Function Chunking | 42,3 (4,0×) | 54,3 (4,3×) | 46,7 (4,5×) |
| LLMLingua | 2,5 (3,6×) | 8,7 (4,1×) | 6,5 (4,3×) |
| LLMLingua-2 | 1,5 (4,6×) | 2,8 (4,6×) | 2,5 (4,6×) |
| LongLLMLingua | 59,3 (3,0×) | 71,3 (4,3×) | 71,2 (5,1×) |
| DietCode (Py+Java saja) | 20,5 (3,7×) | 26,0 (3,7×) | 24,0 (3,7×) |
| SlimCode (Py+Java saja) | 25,0 (4,3×) | 34,0 (4,3×) | 37,5 (4,3×) |
| LongCodeZip | 75,3 (5,3×) | 87,2 (4,5×) | 80,7 (5,3×) |

Untuk model tertutup (Tabel V), No Compression → LongCodeZip: Claude-3.7-Sonnet 89,7 → 88,9 dan GPT-4o 87,8 → 88,9, keduanya pada 5,1×. Teks paper menyebut 90,7 untuk Claude, tidak sama dengan tabel. Inkonsistensi ini dicatat di DECISIONS.

Pelajaran untuk desain kita:

1. Kompresi bisa menaikkan skor, bukan hanya menghemat token. Contohnya DeepSeek 38,3 → 75,3. Model lemah tampaknya terganggu oleh konteks panjang ("lost in the middle").
2. Pesaing utama kita adalah pemeringkatan fungsi. Yang berbeda hanya sinyal peringkatnya: AMI (perplexity), BM25, embedding, atau LLM yang memilih dari indeks.
3. Hal-hal berikut tidak disebut di paper dan menjadi TODO di DECISIONS:
   - ukuran jendela dan stride RAG Sliding Window;
   - top-k dan urutan potongan RAG;
   - rasio target untuk Random dan keluarga LLMLingua;
   - model kompresor LLMLingua dan varian model LLMLingua-2;
   - parameter LongLLMLingua;
   - definisi q (instruksi) LongCodeZip untuk RepoQA;
   - chunker fungsi untuk bahasa selain Python;
   - protokol "10 pengulangan";
   - tokenizer untuk panjang konteks dan rasio.
4. Hipotesis soal angka 11,5K: paper memakai jendela default 16K CodeLlama, tetapi mengukur panjangnya dengan tokenizer model generator. Dengan tokenizer Qwen hasilnya ±12,4K, cukup dekat dengan 11,5K. Hipotesis ini belum terkonfirmasi.

## 3. Baseline

Semua baseline yang punya anggaran dijalankan pada anggaran token yang sama dengan metode kita. Usulannya B ∈ {1K, 2K, 4K} token CodeLlama. Angka ini pilihan kita, bukan dari paper, dan masih perlu disetujui. Dengan begitu kita bisa menggambar kurva skor terhadap token, tidak hanya membandingkan satu titik.

| Baseline | Cara implementasi | Status | Kebutuhan di lab | Risiko utama |
|---|---|---|---|---|
| No Compression (`full`) | `code_context` dari `make_code_context` tanpa perubahan. Prompt memakai TEMPLATE asli. n=1, temperature 0, `max_new_tokens` 1024, tanpa system message. | langsung | Ollama dengan `num_ctx` ≥ 24.576 | Identitas dengan RepoQA asli harus dibuktikan tes (§5.8). Kuantisasi Ollama membuat angka tidak sama dengan paper. |
| No Context (`no_context`) | `code_context = ""`, sehingga prompt = INSTRUCTION + deskripsi + INSTRUCTION. | langsung | – | Seharusnya ≈ 0. Jika > 0, ada bocoran atau model hafal repo populer (kontaminasi data). Berguna sebagai pemeriksa kewarasan. |
| Random Line (`random_line`) | Hapus baris acak (seed per tugas) sampai ≤ B. Urutan baris dipertahankan. Header `Path:` dikeluarkan dari undian (keputusan kita, TODO). | replikasi | – | Paper tidak menyebut target rasionya (yang tercapai 3,5×), jadi dijalankan pada B yang sama dengan metode kita. |
| Random Token (opsional, `random_token`) | Hapus token CodeLlama acak sampai ≤ B, lalu decode. | replikasi | – | Decode token yang terpisah-pisah menghasilkan teks aneh. Skor diperkirakan ≈ 0. |
| RAG Function Chunking (`rag_fn_bm25`, `rag_fn_unixcoder`) | Jendela dipecah menjadi unit fungsi (parser tree-sitter umum); sisa kode menjadi potongan "lain". Skor tiap unit terhadap deskripsi, ambil teratas sampai B, lalu susun kembali dalam urutan asli dengan header Path dan `...` di celah. | pakai library: `rank_bm25` atau BM25 sendiri ±40 baris (CPU); `microsoft/unixcoder-base` (replikasi paper) | UniXcoder di GPU | BM25 menyimpang dari paper, jadi keduanya dijalankan dan dilaporkan terpisah. Deskripsi sengaja tanpa nama, sehingga ada celah kosakata. Fungsi di tepi jendela bisa terpotong. Urutan (asli atau menurut peringkat) tidak disebut paper. |
| RAG Sliding Window (`rag_sw_bm25`, `rag_sw_unixcoder`) | Jendela token yang tumpang tindih diberi skor dengan pemeringkat yang sama; ambil teratas sampai B; gabungkan jendela yang saling tumpang tindih. | replikasi + library | idem | Ukuran dan stride tidak disebut paper (TODO). Usulan kita adalah sapuan {256, 512} token dengan tumpang tindih 50%. Jendela bisa memotong needle (p90 432 token, maks 924). |
| LLMLingua (`llmlingua`) | `llmlingua.PromptCompressor` dengan LM kausal HF kecil, `compress_prompt(code_context, target_token=B)`. | pakai library | HF transformers + GPU | Merusak sintaks, skor diperkirakan ≈ 0 (paper 2,5–8,7). Model kompresor tidak disebut paper (TODO). Versi `transformers` bisa bentrok dengan RepoQA. |
| LLMLingua-2 (`llmlingua2`) | `PromptCompressor(model_name=<llmlingua-2 xlm-roberta-large atau bert-base>, use_llmlingua2=True)`. | pakai library | GPU | Varian model tidak disebut paper (TODO). Risiko sama dengan LLMLingua. |
| LongLLMLingua (`longllmlingua`) | `compress_prompt(context=[unit...], question=deskripsi, rank_method="longllmlingua", ...)`. | pakai library | GPU | Granularitas konteks (fungsi atau file) dan parameternya TODO. Ini yang terkuat di keluarga LLMLingua (59–71 di paper). Komputasinya berat. |
| LongCodeZip (`longcodezip`, belakangan) | Pakai kode resmi (`github.com/YerbaPage/LongCodeZip`) jika mendukung 6 bahasa. Jika tidak, replikasi tahap kasar: unit fungsi → AMI dengan model HF → greedy sampai B = 2K → placeholder. Karena R_fine = 1,0, tahap halus tidak dipakai. | library atau replikasi | HF + GPU (perlu logits; Ollama tidak cukup) | q untuk RepoQA tidak didefinisikan (TODO). Chunker multi-bahasa. Biaya kompresi = N unit × satu forward pass. |
| DietCode, SlimCode | Tidak direplikasi. Angka Tabel IV disalin (hanya Python dan Java) dengan catatan bahwa setup-nya berbeda. | salin angka paper | – | Tidak bisa dibandingkan langsung: model, presisi, dan tokenizer berbeda. DietCode bergantung pada atensi CodeBERT dan dirancang untuk fungsi pendek. |

Jika waktu tidak cukup, urutan pengerjaan baseline: `full`, `no_context`, `rag_fn_bm25`, `rag_sw_bm25`, `random_line`, `longllmlingua`, `longcodezip`, varian UniXcoder, `llmlingua`, `llmlingua2`, `random_token`.

## 4. Varian metode

Prinsip yang diturunkan dari §1:

- (a) Needle harus utuh di konteks akhir. Unit seleksi adalah fungsi utuh, disalin verbatim.
- (b) Skor dapat dipecah menjadi recall tahap seleksi × ketepatan model memilih × ketepatan menyalin. Recall bisa diukur di laptop tanpa LLM.
- (c) Prompt jawaban tetap TEMPLATE RepoQA; yang berubah hanya `code_context`. Satu-satunya pengecualian adalah S8.
- (d) Semua token dihitung, termasuk panggilan LLM tambahan.
- (e) Parameter seperti k, M, dan ambang dituning di split dev, bukan di split test.

Perkiraan biaya di bawah memakai token CodeLlama dan statistik §1.6. Untuk C++/TS jumlah fungsinya kurang terhitung, jadi biaya indeks sebenarnya bisa 1,5–2× lebih besar. Sebagai pembanding, prompt `full` ±16,6K token masuk dalam 1 panggilan.

| # | Varian | Panggilan LLM | Token masuk per tugas (perkiraan) | Rasio vs full | Batas atas recall dari pilot |
|---|---|---|---|---|---|
| S1 | `struct_bm25` (+ kerangka opsional) | 1 | 1,3–2,7K | 6–13× | @5: 0,88, @10: 0,94 |
| S2 | `index_select` (X1-sig / X1-doc) | 2 | 1,9–2,7K | 6–9× | belum diketahui; BM25 signature @10: 0,87 sebagai pembanding |
| S3 | `cascade` (S1 → X1 pada M=20) | 2 | ±1,5K | ±11× | @20: 0,97 |
| S4 | `adaptive_k` (X3, tambahan) | +0 | turun dari S1/S2/S3 | naik | sama dengan dasarnya |
| S5 | `hyde_code` | 2 | ±1,5–1,8K (+150–300 token keluar) | 9–11× | belum diketahui |
| S6 | `pair_verify` (X4, tambahan) | +1 pada tugas ambigu | +±0,65K per perbandingan | turun sedikit | sama dengan dasarnya |
| S7 | `graph_neighbors` (X2, tambahan) | +0 | konteks 2–3× dari X1 | 3–5× | naik sedikit |
| S8 | `pointer` (pengubah, jalur terpisah) | −1 (tanpa panggilan baca) | ±1,0–1,8K, keluaran ±5 token | 9–16× | sama dengan dasarnya |

Urutan prioritas: S1 → S2 → S3 → S4 → S5 → S6 → S7 → S8.

- S1 bisa diuji penuh di laptop lewat retensi needle, dan menjadi lantai bagi semua varian lain.
- S2 adalah inti usulan.
- S3 menekan biaya S2.
- S4–S6 adalah tambahan yang murah.
- S7 paling mahal dibangun dengan manfaat yang paling tidak pasti.
- S8 butuh keputusan validitas.

### S1. `struct_bm25`: BM25 terstruktur + kerangka (tanpa LLM tambahan)

- **Cara kerja.** Jendela dipecah menjadi unit fungsi (termasuk method, konstruktor, arrow function, fungsi C++ berkualifikasi). Setiap unit diindeks per field: nama (dipecah camelCase/snake_case), signature, docstring/komentar, dan isi. Skor BM25 per field digabung (bobot atau RRF). Ambil top-k unit utuh, salin verbatim, susun dalam urutan asli dengan header Path. Opsi `skeleton`: unit yang dibuang diganti satu baris signature + `...`, sehingga struktur file tetap terlihat.
- **Kenapa bisa menang.** Pilot menunjukkan recall@10 = 94% pada ±2K token konteks, setara anggaran LongCodeZip, tanpa biaya GPU untuk kompresi. Konteks yang pendek juga mengurangi efek lost-in-the-middle.
- **Biaya token.** Dengan k=5: ±1,0K token konteks, ditambah 0,6–1,4K jika kerangka dipakai, ditambah 0,24K instruksi dan deskripsi. Totalnya 1,3–2,7K dalam 1 panggilan.
- **Risiko.** Celah kosakata (deskripsi tanpa nama). Recall Rust dan Java lebih rendah (Rust @10 = 84%). Kerangka bisa mendorong model memilih signature yang isinya tidak ada. Bedanya dengan baseline `rag_fn_bm25` (field, kerangka, k adaptif) harus jelas dan diukur lewat ablasi.

### S2. `index_select` (X1): indeks → LLM memilih → baca kandidat

- **Cara kerja.**
  - Panggilan 1: daftar unit bernomor, misalnya `[17] src/black/linegen.py · class LineGenerator · def visit_atom(self, node: Node) -> Iterator[Line]`, ditambah deskripsi. LLM diminta mengembalikan ≤ k ID dalam format JSON.
  - Panggilan 2: prompt RepoQA asli dengan `code_context` berisi unit terpilih saja.
  - Varian indeks:
    - X1-sig: signature saja, tanpa biaya tambahan.
    - X1-doc: signature + baris pertama docstring/komentar, tanpa biaya tambahan.
    - X1-sum: ringkasan satu kalimat oleh LLM. Lihat catatan biaya di bawah.
- **Kenapa bisa menang.** LLM mencocokkan makna nama dan signature dengan deskripsi lebih baik daripada pencocokan kata. Contohnya "combine adjacent strings" ↔ `_merge_string_group`. BM25 yang hanya melihat signature mencapai recall @10 = 87%; LLM diharapkan lebih tinggi, dan itulah yang akan diuji.
- **Biaya token.**
  - Panggilan 1: ±1,0–1,8K token masuk, ±30 token keluar.
  - Panggilan 2 (k=3): ±0,85K token masuk.
  - Total ±1,9–2,7K token masuk.
  - X1-sum: membuat ringkasan berarti membaca semua kode sekali. Biaya per tugas ≥ biaya `full`, kecuali ringkasan dibuat sekali per repo lalu diamortisasi ke banyak tugas. Jika X1-sum dipakai, biaya indeks dilaporkan terpisah.
- **Risiko.** Signature tidak memuat detail "Procedure" dari deskripsi. LLM kecil bisa gagal mengikuti format JSON, jadi perlu parser yang toleran dan fallback ke BM25. Indeks untuk Java dan Python panjang (p90 > 1,3K token). Posisi unit dalam daftar bisa memengaruhi pilihan LLM.

### S3. `cascade`: BM25 menyaring, X1 memutuskan

- **Cara kerja.** S1 memilih M kandidat (mis. 20). X1 hanya melihat indeks M kandidat itu, lalu kode k kandidat terpilih dibaca.
- **Kenapa bisa menang.** Biaya indeks terpotong. Pilot recall@20 = 97%, jadi kerugian dari penyaringan kecil. LLM tinggal menyusun ulang kandidat yang lolos.
- **Biaya token.** ±0,6K (indeks 20 unit) + ±0,85K (baca k=3) ≈ 1,5K token masuk.
- **Risiko.** Recall BM25 menjadi batas atas. Ada dua sumber kesalahan yang berantai.

### S4. `adaptive_k` (X3): jumlah kandidat adaptif

- **Cara kerja.** k ditentukan per tugas dengan salah satu cara:
  - (a) selisih skor BM25 antara peringkat 1 dan 2;
  - (b) jumlah unit di jendela;
  - (c) LLM di S2/S3 mengembalikan ID beserta tingkat keyakinan (tinggi/sedang/rendah), yang dipetakan ke k = 1/3/6.
- **Kenapa bisa menang.** Banyak tugas sebenarnya mudah (BM25 saja sudah tepat di peringkat 1 pada 66% tugas), sehingga k kecil sudah cukup. Tugas sulit mendapat lebih banyak kandidat.
- **Biaya token.** Hampir tidak ada tambahan, dan rata-rata konteks turun.
- **Risiko.** Keyakinan verbal LLM kecil sering tidak terkalibrasi. Ambang harus dituning di dev.

### S5. `hyde_code`: tebakan kode sebagai kueri

- **Cara kerja.** Satu panggilan murah dari deskripsi saja (tanpa konteks): LLM menulis tebakan signature dan isi fungsi. Tebakan itu menjadi kueri BM25 kode-ke-kode, digabung dengan kueri deskripsi lewat RRF. Langkah berikutnya sama dengan S1 atau S3.
- **Kenapa bisa menang.** Menutup celah kosakata. Nama dan API yang ditebak (mis. `merge`, `string`, `leaf`) cenderung muncul di kode asli.
- **Biaya token.** Tambahan ±0,24K token masuk dan 150–300 token keluar. Token keluar lebih lambat dan lebih mahal daripada token masuk.
- **Risiko.** Nama hasil halusinasi bisa menyesatkan. Latensi bertambah.

### S6. `pair_verify` (X4): LLM membandingkan dua kandidat teratas

- **Cara kerja.** Dijalankan hanya jika kandidat 1 dan 2 berdekatan (selisih skor kecil atau keyakinan rendah). LLM melihat kode lengkap keduanya bersama deskripsi, lalu memilih A atau B. Jika perlu, perbandingan dijalankan dua kali dengan urutan dibalik untuk mengurangi bias posisi. Prompt jawaban berisi pemenang di posisi pertama.
- **Kenapa bisa menang.** Membedakan fungsi saudara yang mirip. Rata-rata ada 1 needle pesaing per jendela.
- **Biaya token.** ±0,65K token masuk per perbandingan (2× jika dua urutan), dan hanya pada tugas yang ambigu.
- **Risiko.** Bias posisi. Jika hanya pemenang yang disimpan dan pilihannya salah, needle hilang dari konteks akhir.

### S7. `graph_neighbors` (X2): tambah tetangga graf

- **Cara kerja.** Bangun graf di dalam jendela: pemanggil dan yang dipanggil (mencocokkan identifier pemanggilan dengan nama unit), serta anggota class yang sama. Tambahkan tetangga 1-hop dari kandidat terpilih sampai anggaran habis.
- **Kenapa bisa menang.** Menyelamatkan recall jika LLM memilih pemanggil atau saudara needle.
- **Biaya token.** Konteks 2–3× lebih besar dari X1.
- **Risiko.** Untuk SNF manfaatnya mungkin kecil, karena tugas hanya butuh satu fungsi. Resolusi nama berbasis teks tidak akurat. Varian ini paling mahal untuk dibangun.

### S8. `pointer`: jawaban berupa penunjuk (pengubah, jalur terpisah)

- **Cara kerja.** Setelah seleksi final, sistem menyalin teks unit terpilih apa adanya sebagai jawaban. LLM hanya perlu menjawab ID.
- **Kenapa bisa menang.** Token keluar turun dari ±200 menjadi ±5. Tidak ada kesalahan salin, dan panggilan baca bisa dihilangkan.
- **Biaya token.** Paling murah dari semua varian.
- **Risiko validitas.** Yang menyalin fungsi bukan model, sehingga bisa dianggap mengubah tugas. Karena itu S8 hanya dilaporkan sebagai jalur "dibantu alat" dan tidak dibandingkan langsung dengan baseline. Butuh keputusanmu atau pembimbing.

## 5. Kerangka kode

### 5.1 Prinsip dan struktur folder

Tidak ada perubahan di `repoqa/` maupun `scripts/` asli. Semua kode baru berada di paket `repoqa_ctx/` (nama bisa diganti) dan di `tests/`.

```text
repoqa_ctx/
  __init__.py
  run.py              # CLI: --strategy, --dry-run, --llm {ollama,fake}
  tasks.py            # membangun tugas identik RepoQA (memanggil make_code_context asli),
                      # ID opak, split dev/test
  codeparse.py        # tree-sitter: pecah code_context menjadi segmen file dan unit
                      # (fungsi/kelas/lain), signature, kerangka; gabung ulang tanpa kehilangan isi
  retrieval.py        # BM25 (+ pemecah identifier), RRF, pembungkus UniXcoder (opsional)
  tokens.py           # penghitung token: tokenizer RepoQA (CodeLlama) + opsional tokenizer model
  llm.py              # LLMClient, OllamaClient, FakeClient, CallLogger
  summarize.py        # tabel skor + token + rasio + uji berpasangan
  strategies/
    __init__.py       # registry: nama -> modul
    base.py           # StrategyInput, StrategyOutput, utilitas perakitan konteks
    full.py
    no_context.py
    random_line.py
    random_token.py
    rag_fn.py         # rag_fn_bm25, rag_fn_unixcoder
    rag_sw.py         # rag_sw_bm25, rag_sw_unixcoder
    llmlingua.py      # llmlingua, llmlingua2, longllmlingua (impor lazy)
    longcodezip.py
    struct_bm25.py    # S1
    index_select.py   # S2 (X1-sig, X1-doc, X1-sum)
    cascade.py        # S3
    hyde_code.py      # S5
    ...               # S4/S6/S7/S8 sebagai opsi atau pembungkus strategi lain
tests/
  fixtures/           # dataset mini sintetis (boleh di-commit)
  test_full_identity.py
  test_codeparse_roundtrip.py
  test_no_leak.py
  test_strategies_dryrun.py
  test_fake_pipeline.py
  test_selection_parser.py
  test_llm_client.py
  test_summarize.py
requirements-ctx.txt  # requests, pytest, (rank_bm25); dependensi lab dipisah
```

### 5.2 Antarmuka strategi

```python
# repoqa_ctx/strategies/base.py
@dataclass(frozen=True)
class StrategyInput:
    uid: str                 # ID opak (hash), BUKAN make_task_id yang memuat nama needle
    language: str
    code_context: str        # jendela asli dari make_code_context
    description: str         # "\nFunction Description:...\n"
    instruction: str         # INSTRUCTION asli
    code_context_size: int
    params: dict             # dari --strategy-args (JSON)
    seed: int                # seed global + hash(uid)

@dataclass
class StrategyOutput:
    code_context: str                  # konteks untuk panggilan jawaban (TEMPLATE asli)
    final_answer: str | None = None    # hanya S8 pointer: jawaban tanpa panggilan baca
    meta: dict = field(default_factory=dict)   # unit terpilih, skor, k, parse error, dll.

def build_context(inp: StrategyInput, llm: LLMClient, tok: TokenCounter) -> StrategyOutput: ...
```

- Strategi tidak pernah menerima `name`, `path`, `needle_token_*`, maupun field `needles` dan `functions` dari dataset.
- Strategi hanya melihat jendela, bukan isi repo lain. Ini menjaga perbandingan tetap adil, karena semua metode memampatkan masukan yang sama.
- Panggilan LLM tambahan dilakukan lewat `llm`, sehingga otomatis tercatat.

### 5.3 Runner dan flag

```bash
python -m repoqa_ctx.run \
  --strategy struct_bm25 --strategy-args '{"k": 5, "skeleton": true}' \
  --model qwen2.5-coder:7b-instruct --llm ollama --ollama-url http://localhost:11434 \
  --num-ctx 24576 --max-new-tokens 1024 --code-context-size 16384 \
  --languages python,go --split test --limit 20 --seed 0 \
  --result-dir results/ctx [--dry-run]
```

Alur per tugas:

1. Bangun tugas seperti RepoQA (memakai ulang cache tugas asli bila ada).
2. Panggil `build_context`.
3. Rakit prompt dengan TEMPLATE asli dan `code_context` baru.
4. Panggil LLM tahap `answer`.
5. Tulis satu baris JSONL.

Run bisa dilanjutkan jika terputus: tugas yang sudah ada di JSONL dilewati, seperti di kode asli.

### 5.4 Mode `--dry-run`

- Menjalankan langkah 1–3, tetapi melewati panggilan jawaban. Tahap tambahan yang butuh LLM (S2, S3, S5) memakai `--llm`, dan default-nya `fake` saat dry-run, jadi tidak perlu model.
- Menulis `dryrun.jsonl` per tugas: token konteks sebelum dan sesudah, token prompt, jumlah unit, unit terpilih, waktu bangun konteks, dan `needle_retained`.
- `needle_retained` dihitung oleh runner, di luar strategi, dengan teks acuan berbasis baris seperti evaluator. Nilainya memberi batas atas skor tanpa LLM.
- Mencetak ringkasan: rata-rata token, rasio, dan retensi per bahasa.
- Dengan `--llm fake --fake-mode oracle`, tahap seleksi menerima jawaban benar. Ini hanya untuk menguji alur, bukan untuk melaporkan hasil.

### 5.5 Pembungkus LLM dan mode palsu

```python
@dataclass
class LLMResult:
    text: str
    stage: str                      # "answer", "select", "verify", "hyde", ...
    prompt_tokens: int              # tokenizer RepoQA (CodeLlama) atas teks yang dikirim
    completion_tokens: int          # idem atas jawaban
    prompt_tokens_backend: int | None       # prompt_eval_count Ollama
    completion_tokens_backend: int | None   # eval_count Ollama
    latency_s: float                # wall clock
    backend_timings: dict           # total/load/prompt_eval/eval duration (ns -> s)

class LLMClient(Protocol):
    def chat(self, user: str, *, stage: str, max_tokens: int,
             system: str | None = None) -> LLMResult: ...
```

- **`OllamaClient`.** `POST /api/chat` dengan `stream=false` dan `options={temperature: 0, seed, num_ctx, num_predict, repeat_penalty: 1.0}`. Timeout dan retry memakai `requests` (bukan SIGALRM, sehingga jalan di Windows dan Linux). Ada satu panggilan pemanasan yang tidak dihitung. Versi Ollama, digest model, dan opsi dicatat di `run_config.json`.
- **`FakeClient(mode)`.** Deterministik, latensi 0, token dihitung dengan cara yang sama. Mode yang tersedia:
  - `oracle`: jawaban benar diberikan lewat jalur samping, khusus tes;
  - `first_block`: mengembalikan fungsi pertama di konteks;
  - `empty`;
  - `scripted`: daftar jawaban yang sudah ditentukan;
  - `select_first`: mengembalikan k ID pertama.
- **`CallLogger`.** Membungkus klien apa pun dan menulis satu baris per panggilan ke `calls.jsonl`, sehingga log klien nyata dan palsu sama persis.
- **Hitungan token.** Hitungan utama memakai tokenizer kita sendiri, karena Ollama bisa memakai ulang cache prefix sehingga `prompt_eval_count` lebih kecil (perlu dicek di lab). Hitungan dari Ollama tetap disimpan sebagai pembanding.

### 5.6 Format log dan hasil

```text
results/ctx/ntoken_16384/<strategi>[-<hash argumen>]/
    <model_aman>.jsonl          # format RepoQA + "strategy", "strategy_args", "ctx_stats", "calls"
    <model_aman>-SCORES.json    # dari compute_score asli (tanpa prompt interaktif)
    <model_aman>.calls.jsonl    # satu baris per panggilan LLM
    run_config.json             # argumen, commit git, versi paket/Ollama, opsi model, seed
    dryrun.jsonl                # bila --dry-run
results/ctx/summary.md, summary.csv
```

- Baris JSONL tetap memuat semua field asli, sehingga `compute_score` asli bisa membacanya.
- `needle_token_start/end` diisi nilai asli dari jendela `full`. `compute_score` hanya menyalinnya ke field `position`.
- `ignore_comments=False` untuk semua strategi, supaya penilaian tidak berubah.

### 5.7 Skrip ringkasan

```bash
python -m repoqa_ctx.summarize results/ctx/ntoken_16384 --baseline full --threshold 0.8
```

Kolom per (strategi, model):

- n tugas
- pass@1 pada ambang 0,8, untuk semua bahasa dan per bahasa
- rata-rata token konteks
- rasio konteks, dihitung dua cara: rasio dari rata-rata dan rata-rata rasio per tugas (TODO: mana yang dipakai paper)
- total token masuk per tugas (semua panggilan)
- token keluar per tugas
- rasio end-to-end, yaitu token masuk `full` dibagi total token masuk strategi
- jumlah panggilan per tugas
- latensi rata-rata, p50, dan p95
- retensi needle
- Δ pass@1 terhadap `full` dengan CI 95% bootstrap berpasangan
- p-value uji McNemar eksak

Keluaran dalam bentuk Markdown dan CSV. Grafik skor terhadap token menyusul di fase berikutnya.

### 5.8 Tes (tanpa LLM, tanpa GPU)

| Tes | Isi |
|---|---|
| `test_full_identity` | Panggil `evaluate_model` asli dengan backend yang tidak dikenal pada dataset mini. Tugas sudah ditulis ke cache ([baris 518-522](../repoqa/search_needle_function.py#L518-L522)) sebelum `ValueError` di [baris 565](../repoqa/search_needle_function.py#L565); `CACHE_DIR` di-monkeypatch ke folder sementara. Bandingkan dengan tugas dari `tasks.py`, lalu pastikan prompt `full` sama byte per byte dengan perakitan TEMPLATE asli. |
| `test_codeparse_roundtrip` | Untuk 6 bahasa: pecah lalu gabung ulang harus menghasilkan `code_context` yang persis sama, termasuk kasus fungsi terpotong di tepi jendela. |
| `test_no_leak` | `StrategyInput` tidak memuat nama, path, atau posisi. Jika nama needle di dataset diganti, keluaran strategi tidak berubah. |
| `test_strategies_dryrun` | Setiap strategi: deterministik untuk seed yang sama, tidak melebihi anggaran (dengan toleransi), dan kerangka/unit tersusun dalam urutan asli. |
| `test_fake_pipeline` | Alur ujung ke ujung: `full` + `oracle` → pass@1 = 1,0 lewat `compute_score` asli; `empty` → 0,0. |
| `test_selection_parser` | JSON valid, daftar dipisah koma, teks bebas, ID di luar rentang, dan fallback. |
| `test_llm_client` | HTTP di-mock: payload memuat `num_ctx` dan `temperature=0`, field `prompt_eval_count` terbaca, retry berjalan. |
| `test_summarize` | Data hasil kecil → tabel, rasio, dan uji berpasangan bernilai benar. |

Dataset mini sintetis (beberapa file kecil per bahasa) boleh di-commit. Tes yang memakai dataset asli diberi penanda `@pytest.mark.dataset` dan dilewati jika cache dataset tidak ada.

### 5.9 Lingkungan

- Laptop: venv Python 3.11 (lihat jebakan #8), `requirements.txt` tanpa `vllm`, ditambah `requirements-ctx.txt`. Tokenizer CodeLlama diunduh otomatis oleh `transformers` (±2 MB, tidak di-gate). Tidak butuh torch atau GPU.
- Lab: tambahan Ollama untuk model generator, dan `transformers` + `torch` + GPU untuk UniXcoder, LLMLingua, dan LongCodeZip (mereka butuh logits atau embedding).

### 5.10 Alternatif: kait minimal di kode asli

Alternatifnya, tambahkan parameter `strategy="full"` ke `evaluate_model` dan ±4 baris setelah [baris 515](../repoqa/search_needle_function.py#L515).

- Kelebihan: loop pembangunan tugas tidak diduplikasi.
- Kekurangan: menyentuh file asli dan tetap butuh pembungkus LLM baru untuk Ollama.

Saya merekomendasikan pendekatan paket terpisah (§5.1), dengan `test_full_identity` sebagai pengaman terhadap duplikasi loop.

## 6. Protokol eksperimen (garis besar)

- **Split.** Dev = 1 repo per bahasa (60 tugas), dipilih acak dengan seed tetap dan dicatat. Test = 540 tugas sisanya. Semua tuning hanya di dev. Hasil utama dilaporkan pada test; untuk perbandingan dengan paper, dilaporkan juga pada 600 tugas.
- **Urutan di lab:**
  1. Uji kewarasan: `full` lewat CLI asli (`--backend openai --base-url http://localhost:11434/v1`, di Linux, `num_ctx` diatur di server) dibandingkan dengan runner kita pada 30 tugas.
  2. `full` dan `no_context` pada 600 tugas.
  3. Baseline BM25 dan S1 pada B ∈ {1K, 2K, 4K}.
  4. S2 dan S3.
  5. Keluarga LLMLingua dan LongCodeZip.
  6. S4–S8 dan ablasi.
- **Pengulangan.** Temperature 0, satu run. Nondeterminisme GPU dicek dengan mengulang `full` pada 50 tugas.
- **Kriteria keberhasilan** (perlu disetujui): pass@1@0,8 pada test tidak lebih rendah dari `full` lebih dari margin tertentu (non-inferiority, dengan CI bootstrap), dengan rasio token end-to-end ≥ target.

## 7. Pertanyaan terbuka

1. `CLAUDE.md` tidak ada di repo. Apakah ada di tempat lain, atau perlu saya buat di fase berikutnya?
2. `referensi/` hanya berisi paper LongCodeZip. Apakah ada paper lain yang ingin dimasukkan (RepoQA, LLMLingua, LongLLMLingua, DietCode, SlimCode)? Beberapa TODO parameter mungkin terjawab dari sana atau dari repo resmi LongCodeZip.
3. Komputer lab: GPU apa, berapa VRAM, OS apa (Linux atau Windows; lihat jebakan #2 dan #5), dan versi Ollama berapa? Apakah HF `transformers` + GPU juga tersedia untuk UniXcoder, LLMLingua, dan LongCodeZip?
4. Model generator: `qwen2.5-coder:7b-instruct`, `deepseek-coder:6.7b-instruct`, atau yang lain? Pakai tag kuantisasi apa (default Q4 atau q8_0/fp16 agar lebih dekat ke paper)?
5. Definisi "sama atau lebih baik" dan "token jauh lebih sedikit": berapa margin non-inferiority yang diterima (mis. −2 poin)? Berapa target rasio end-to-end (mis. ≥ 4× atau ≥ 5×)? Apakah token keluar dan token model kecil dihitung setara dengan token model utama?
6. Apakah split dev/test seperti di §6 disetujui?
7. Apakah S8 (jawaban penunjuk) boleh dilaporkan sebagai jalur terpisah?
8. Apakah X1-sum (indeks ringkasan LLM yang dibuat sekali per repo dan diamortisasi) dianggap setting yang sah?
9. Ukuran konteks: hanya 16.384, atau juga 32K/64K? Kompresi makin bernilai pada konteks yang lebih panjang, tetapi model dan VRAM di lab membatasi.
10. RAG: jalankan BM25 (permintaanmu) dan UniXcoder (paper) dua-duanya? Saya merekomendasikan dua-duanya.
11. Nama dan lokasi paket: `repoqa_ctx/strategies/` atau `strategies/` di root?
12. Bolehkan saya membuat venv Python 3.11 di laptop?
13. Apakah `results/ctx/` perlu ditambahkan ke `.gitignore`? Saat ini `*.jsonl` dan `results/*.json` sudah diabaikan, tetapi `summary.md` dan `summary.csv` belum. Catatan: `.gitignore` sedang punya perubahan darimu yang belum di-commit (`referensi/`).
14. Apakah 54 tugas dengan offset byte yang meleset perlu dianalisis terpisah, atau cukup dicatat?
