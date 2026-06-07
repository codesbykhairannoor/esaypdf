# EsayPDF (Autonomous AI Telegram Bot)

Bot Telegram yang menggunakan AI (DeepSeek) untuk secara otomatis membaca panduan lomba dalam format PDF, meriset data via Google, dan menulis essay yang langsung diekspor ke dalam file `.docx`.

## Fitur Utama
- **PDF Reader**: Ekstraksi rubrik dan panduan langsung dari file PDF lomba.
- **DeepSeek AI**: Menggunakan model `deepseek-chat` untuk kualitas penalaran tinggi dan hemat token.
- **RAG & Memory (ChromaDB)**: AI dapat mengingat teguran atau "kritik" dari pengguna sehingga tidak mengulangi kesalahan di essay berikutnya.
- **Web Search**: Integrasi DuckDuckGo Search untuk mencari data statistik terbaru.

## Instalasi
1. Clone repositori ini.
2. Buat file `.env` berdasarkan `.env.example` dan masukkan Token Telegram serta API Key DeepSeek Anda.
3. Jalankan `pip install -r requirements.txt`.
4. Jalankan `python bot.py`.
