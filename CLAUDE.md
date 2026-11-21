# Konteks proyek

Riset: manajemen konteks LLM untuk kode panjang (P4). Tugas: RepoQA Search Needle Function.
Tujuan: skor pass@1 (BLEU>=0.8) sama atau lebih baik dari baseline dengan token jauh lebih sedikit.
Ambang kemiripan: 0,8 (bawaan RepoQA dan paper LongCodeZip) sebagai angka utama; semua ambang 0,0-1,0
tetap dicatat dan dilaporkan sebagai uji ketahanan.
Pertanyaan inti: apakah seleksi fungsi yang melibatkan LLM mengalahkan seleksi murah tanpa LLM
(BM25 terstruktur) pada anggaran token yang sama.

# Lingkungan

- Laptop: hanya tulis kode dan tes tanpa LLM/GPU. Dilarang eksperimen penuh dan unduh model besar.
- Lab: Ollama + GPU, tempat semua run sungguhan. Alur: commit di laptop -> push -> clone di lab.
- Python 3.11 (tree-sitter tidak punya wheel untuk 3.13).

# Aturan

- Jangan ubah kode asli repoqa/ dan scripts/. Kode baru di repoqa_ctx/ dan tests/.
- Strategi full harus identik dengan RepoQA asli (dibuktikan tes). Jangan ubah penilaian bawaan.
- Strategi hanya menerima ID opak, bahasa (language), jendela kode, dan deskripsi, ditambah
  konfigurasi run (budget, params, seed). Dilarang memakai nama, path, posisi needle, atau
  pengetahuan konstruksi benchmark (FUNCTION_QUERY, kriteria needle).
- Anggaran token = tokenizer CodeLlama RepoQA. Semua panggilan LLM tambahan dihitung sebagai token.
- Tuning hanya di split dev; laporan di split test. Split dibuat dengan seed tetap dan di-commit.
- Jangan mengarang parameter paper. Tidak yakin -> tulis TODO di docs/DECISIONS.md.
- Commit kecil dan sering. Jangan push. Jangan commit results/ atau cache dataset.
- Ringkasan hasil di reports/ (di-commit). Raw hasil di results/ (diabaikan).

# Bahasa

Komentar dan nama kode dalam bahasa Inggris. Dokumen dalam bahasa Indonesia.

# Dokumen

docs/PLAN.md (rencana), docs/DECISIONS.md (keputusan dan TODO), third_party/ (hanya baca, diabaikan git).

- Lab: RTX 3060 12GB VRAM, i7-12700K, RAM 32GB. Generator 7B terkuantisasi lewat Ollama.
  Jangan asumsikan model 7B fp16 muat di GPU. OS dan versi Ollama: cek dengan scripts/lab_check.py.
