import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import warnings

# Mengabaikan peringatan yang tidak relevan
warnings.filterwarnings("ignore", category=UserWarning, message="Given custom collection metadata")

# Muat variabel lingkungan (OPENAI_API_KEY) dari file .env
load_dotenv()

# Tentukan path tempat database vektor Anda disimpan
CHROMA_DB_DIR = "db_chroma"

def main():
    """
    Fungsi utama untuk menjalankan RAG percakapan (history-aware).
    """
    print("--- Memulai Chatbot RAG ---")

    # --- 1. INISIALISASI KOMPONEN ---
    
    # Pastikan database Chroma sudah ada
    if not os.path.exists(CHROMA_DB_DIR):
        print(f"Error: Direktori database '{CHROMA_DB_DIR}' tidak ditemukan.")
        print("Harap jalankan 'injection_pipeline.py' terlebih dahulu.")
        return

    # Inisialisasi model embedding (HARUS SAMA dengan saat injeksi)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Muat Vector Store yang sudah ada
    print(f"Memuat vector store dari '{CHROMA_DB_DIR}'...")
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Inisialisasi retriever (untuk mengambil dokumen)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Inisialisasi model chat (LLM untuk reformulasi dan menjawab)
    chat_model = ChatOpenAI(model="gpt-3.5-turbo")

    # Siapkan list untuk menyimpan riwayat percakapan
    chat_history = []

    print("--- Siap! ---")
    print("Tanyakan apa saja tentang dokumen Anda. Ketik 'quit' untuk keluar.")

    # --- 2. LOOP PERCAKAPAN ---
    while True:
        try:
            # Dapatkan input dari pengguna
            user_query = input("\nAnda: ")
            if user_query.lower() == 'quit':
                print("--- Sampai jumpa! ---")
                break

            # --- 3. LANGKAH 1: REFORMULASI PERTANYAAN (jika perlu) ---
            reformulated_query = user_query
            
            if chat_history:
                print("--- Mengecek riwayat untuk reformulasi...")
                
                # Buat prompt untuk mereformulasi pertanyaan
                reformulation_prompt_messages = [
                    SystemMessage(content="Anda adalah asisten yang ahli dalam menulis ulang pertanyaan. "
                                        "Lihat riwayat obrolan dan pertanyaan lanjutan dari pengguna. "
                                        "Tulis ulang pertanyaan lanjutan tersebut menjadi pertanyaan yang lengkap dan berdiri sendiri (standalone) "
                                        "yang dapat dimengerti tanpa perlu melihat riwayat obrolan. "
                                        "HANYA kembalikan pertanyaan yang sudah ditulis ulang."),
                    *chat_history, # Perluas riwayat obrolan di sini
                    HumanMessage(content=user_query)
                ]
                
                # Panggil LLM untuk reformulasi
                response = chat_model.invoke(reformulation_prompt_messages)
                reformulated_query = response.content
                print(f"--- Pertanyaan direformulasi: {reformulated_query} ---")

            # --- 4. LANGKAH 2: RETRIEVAL (PENGAMBILAN DATA) ---
            print("--- Mengambil dokumen relevan...")
            # Gunakan pertanyaan yang sudah direformulasi untuk mencari dokumen
            relevant_docs = retriever.invoke(reformulated_query)
            
            if not relevant_docs:
                print("--- Tidak ada dokumen relevan yang ditemukan. ---")
                continue

            # Format dokumen yang relevan menjadi satu string
            context_str = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

            # --- 5. LANGKAH 3: GENERASI JAWABAN (DENGAN GROUNDING) ---
            print("--- Menghasilkan jawaban... ---")

            # Buat prompt untuk generasi jawaban
            # Kita masukkan riwayat obrolan DAN dokumen relevan
            generation_prompt_messages = [
                SystemMessage(content="Anda adalah asisten yang membantu menjawab pertanyaan. "
                                    "Jawab pertanyaan pengguna HANYA berdasarkan konteks dokumen yang disediakan. "
                                    "Jika jawaban tidak ada dalam konteks, katakan 'Maaf, saya tidak menemukan informasi tersebut di dokumen.' "
                                    "Jangan gunakan pengetahuan di luar dokumen."
                                    f"\n\n--- KONTEKS DOKUMEN ---\n{context_str}"),
                *chat_history,
                HumanMessage(content=user_query) # Gunakan pertanyaan ASLI pengguna
            ]

            # Panggil LLM untuk menghasilkan jawaban
            ai_response = chat_model.invoke(generation_prompt_messages)
            answer = ai_response.content

            print(f"\nAI: {answer}")

            # --- 6. LANGKAH 4: PERBARUI RIWAYAT ---
            # Simpan pertanyaan ASLI dan jawaban AI ke riwayat
            chat_history.append(HumanMessage(content=user_query))
            chat_history.append(AIMessage(content=answer))

        except Exception as e:
            print(f"\nTerjadi error: {e}")
            if "insufficient_quota" in str(e):
                print("Error: Kuota OpenAI Anda habis. Harap cek tagihan Anda.")
                break
            continue

# Menjalankan fungsi main saat script dieksekusi
if __name__ == "__main__":
    main()