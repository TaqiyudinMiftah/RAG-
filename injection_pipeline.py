import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Muat variabel lingkungan (OPENAI_API_KEY) dari file .env
load_dotenv()

# Tentukan path untuk data sumber dan database vektor
DOCS_DIR = "docs"
CHROMA_DB_DIR = "db_chroma"

def load_documents(directory_path: str):
    """
    Memuat dokumen .txt dari direktori yang ditentukan.
    """
    print(f"Memuat dokumen dari {directory_path}...")
    
    # Mengecek apakah direktori ada
    if not os.path.exists(directory_path):
        print(f"Error: Direktori '{directory_path}' tidak ditemukan.")
        return []

    # Tentukan argumen keyword untuk TextLoader
    loader_kwargs = {'encoding': 'utf-8'}

    # Menggunakan DirectoryLoader untuk memuat file .txt
    loader = DirectoryLoader(
        directory_path,
        glob="*.txt",  # Hanya memuat file dengan ekstensi .txt
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True,
        loader_kwargs=loader_kwargs  # <-- TAMBAHKAN BARIS INI
    )
    
    try:
        documents = loader.load()
    except Exception as e:
        print(f"Terjadi error saat memuat dokumen: {e}")
        print("Pastikan file .txt Anda disimpan dalam format UTF-8.")
        return []
    
    if not documents:
        print("Tidak ada file .txt yang ditemukan di direktori.")
        return []
        
    print(f"Berhasil memuat {len(documents)} dokumen.")
    
    return documents

def split_documents(documents: list, chunk_size: int = 800, chunk_overlap: int = 0):
    """
    Memecah dokumen menjadi chunks yang lebih kecil menggunakan metode rekursif.
    """
    print(f"Memecah {len(documents)} dokumen menjadi chunks (size={chunk_size}, overlap={chunk_overlap})...")
    
    # Gunakan RecursiveCharacterTextSplitter, BUKAN CharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Ini adalah pemisah default, tapi bagus untuk ditulis agar jelas
        separators=["\n\n", "\n", " ", ""],
        strip_whitespace=True
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        print("Gagal memecah dokumen.")
        return []

    print(f"Total chunks yang dihasilkan: {len(chunks)}")
    # print("--- Contoh Chunk ---")
    # print(chunks[0].page_content)
    # print("Metadata:", chunks[0].metadata)
    # print("--------------------")
    
    return chunks

def embed_and_store(chunks: list, persist_directory: str):
    """
    Membuat embeddings dari chunks dan menyimpannya ke ChromaDB.
    """
    print(f"Membuat embeddings dan menyimpan ke ChromaDB di '{persist_directory}'...")
    
    # 1. Inisialisasi model embedding
    # Menggunakan model 'text-embedding-3-small' dari OpenAI
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 2. Membuat dan menyimpan vector store
    # Chroma.from_documents akan:
    # - Mengambil setiap chunk
    # - Membuat embedding-nya menggunakan embedding_model
    # - Menyimpannya ke persist_directory
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"} # Menentukan algoritma similarity
        )
        
        print(f"Vector store berhasil dibuat dan disimpan di '{persist_directory}'.")
        return vector_store
        
    except Exception as e:
        print(f"Terjadi error saat membuat vector store: {e}")
        return None

def main():
    """
    Fungsi utama untuk menjalankan seluruh pipeline injeksi data.
    """
    print("--- Memulai Injection Pipeline RAG ---")
    
    # Langkah 1: Muat Dokumen
    documents = load_documents(DOCS_DIR)
    if not documents:
        print("Pipeline dihentikan karena tidak ada dokumen untuk diproses.")
        return

    # Langkah 2: Pecah Dokumen (Chunking)
    chunks = split_documents(documents, chunk_size=800, chunk_overlap=0)
    if not chunks:
        print("Pipeline dihentikan karena gagal melakukan chunking.")
        return
        
    # Langkah 3: Buat Embedding dan Simpan
    vector_store = embed_and_store(chunks, CHROMA_DB_DIR)
    if not vector_store:
        print("Pipeline dihentikan karena gagal membuat vector store.")
        return

    print("--- Injection Pipeline RAG Selesai ---")
    print(f"Total dokumen diproses: {len(documents)}")
    print(f"Total chunks disimpan: {len(chunks)}")
    print(f"Database vektor disimpan di: {CHROMA_DB_DIR}")

# Menjalankan fungsi main saat script dieksekusi
if __name__ == "__main__":
    main()