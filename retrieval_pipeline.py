import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import warnings

# Mengabaikan peringatan spesifik dari ChromaDB yang tidak relevan
warnings.filterwarnings("ignore", category=UserWarning, message="Given custom collection metadata")

# Muat variabel lingkungan (OPENAI_API_KEY) dari file .env
load_dotenv()

# Tentukan path tempat database vektor Anda disimpan
CHROMA_DB_DIR = "db_chroma"

def main():
    """
    Fungsi utama untuk menjalankan pipeline pengambilan data (retrieval).
    """
    print("--- Memulai Retrieval Pipeline RAG ---")

    # Pastikan database Chroma sudah ada
    if not os.path.exists(CHROMA_DB_DIR):
        print(f"Error: Direktori database '{CHROMA_DB_DIR}' tidak ditemukan.")
        print("Harap jalankan 'injection_pipeline.py' terlebih dahulu.")
        return

    # 1. Inisialisasi model embedding
    # PENTING: Gunakan model yang SAMA PERSIS dengan saat injeksi data
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. Muat Vector Store yang sudah ada
    print(f"Memuat vector store dari '{CHROMA_DB_DIR}'...")
    try:
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"} # Tentukan algoritma similarity
        )
    except Exception as e:
        print(f"Terjadi error saat memuat vector store: {e}")
        return

    # 3. Tentukan pertanyaan (query) Anda
    # Ganti pertanyaan ini untuk menguji dokumen Anda
    query = "In what year did Tesla begin production of the Roadster?"
    # query = "What was Nvidia's first graphics accelerator called?"
    # query = "How much did Microsoft pay to acquire GitHub?"
    
    print(f"\nMelakukan pencarian untuk query: '{query}'")

    # 4. Buat retriever
    # 'k=5' berarti kita ingin mengambil 5 chunks teratas yang paling relevan
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # 5. Lakukan pencarian (invoke retriever)
        # Ini akan:
        # - Meng-embed query Anda
        # - Membandingkannya dengan semua chunks di database
        # - Mengembalikan 'k' chunks yang paling mirip
        relevant_docs = retriever.invoke(query)

        # 6. Tampilkan hasil
        print("\n--- Hasil Dokumen yang Relevan ---")
        if not relevant_docs:
            print("Tidak ada dokumen relevan yang ditemukan.")
            return

        for i, doc in enumerate(relevant_docs):
            print(f"\n--- Dokumen Relevan #{i + 1} ---")
            
            # Membersihkan teks untuk tampilan yang lebih rapi
            page_content_cleaned = ' '.join(doc.page_content.split())
            print(f"Isi Teks: {page_content_cleaned[:500]}...") # Tampilkan 500 karakter pertama
            
            # Tampilkan metadata (sumber file)
            source = doc.metadata.get('source', 'Tidak diketahui')
            print(f"Sumber: {source}")

    except Exception as e:
        print(f"Terjadi error saat melakukan retrieval: {e}")
        # Error ini bisa terjadi jika API key Anda salah/habis kuota saat meng-embed query
        if "insufficient_quota" in str(e):
             print("Error: Kuota OpenAI Anda habis. Harap cek tagihan Anda.")

    print("\n--- Retrieval Pipeline Selesai ---")

# Menjalankan fungsi main saat script dieksekusi
if __name__ == "__main__":
    main()