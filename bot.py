import os
import uuid
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from agent import EssayAgent

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Initialize Agent
agent = EssayAgent()

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    welcome_msg = (
        "Halo bos! Gw AI Agent penulis essay lu.\n"
        "Kirim file PDF panduan lomba ke sini, tambahkan caption/pesan "
        "kalo ada instruksi khusus (misal tema atau deadline).\n\n"
        "Kalo mau ngasih kritik buat bahan belajar gw, ketik /critique [kritik lu]."
    )
    await update.message.reply_text(welcome_msg)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming PDF files."""
    document = update.message.document
    
    if not document.file_name.lower().endswith('.pdf'):
        await update.message.reply_text("Maaf bro, gw cuma nerima file PDF untuk panduan lomba.")
        return
        
    await update.message.reply_text("PDF diterima! Sedang mendownload dan membaca panduan...")
    
    # Check file size (Telegram limits bot downloads to 20MB)
    if document.file_size and document.file_size > 20 * 1024 * 1024:
        await update.message.reply_text("Waduh bro, ukuran file PDF lu kegedean (Maksimal 20MB dari sananya Telegram). Coba kompres dulu PDF-nya di ilovepdf.com atau sejenisnya, terus kirim ulang ke sini!")
        return
        
    try:
        # Download file
        file = await context.bot.get_file(document.file_id)
        pdf_path = os.path.join("data", f"{document.file_id}.pdf")
        await file.download_to_drive(pdf_path)
    except Exception as e:
        await update.message.reply_text(f"Gagal download PDF: {e}")
        return
    
    # Get user prompt from caption
    user_prompt = update.message.caption if update.message.caption else "Buatkan essay berdasarkan panduan di PDF ini."
    
    await update.message.reply_text("Panduan sedang dianalisis. AI sedang meriset dan menulis essay. Proses ini mungkin memakan waktu beberapa menit. Tunggu ya...")
    
    # Output path
    output_filename = f"essay_result_{document.file_id}.docx"
    output_path = os.path.join("data", output_filename)
    
    # Process Essay
    try:
        result_msg = agent.process_essay_request(user_prompt, pdf_path, output_path)
        
        # Send back the DOCX
        if os.path.exists(output_path):
            await update.message.reply_document(
                document=open(output_path, 'rb'),
                caption="Ini hasil essay lu bro. Kalau ada yang kurang pas, kritik aja pakai command /critique [pesan]."
            )
        else:
            await update.message.reply_text("Gagal membuat file DOCX.")
            
    except Exception as e:
        await update.message.reply_text(f"Waduh ada error waktu mikir: {e}")

async def handle_critique(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles learning critiques from user."""
    if not context.args:
        await update.message.reply_text("Cara pakai: /critique [kritik lu]. Contoh: /critique Selalu gunakan bahasa baku.")
        return
        
    critique_text = " ".join(context.args)
    critique_id = str(uuid.uuid4())
    
    response = agent.learn_from_critique(critique_id, critique_text)
    await update.message.reply_text(f"Noted bro! {response}")

def main() -> None:
    """Start the bot."""
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "GANTI_DENGAN_TOKEN_BOT_TELEGRAM_ANDA":
        print("ERROR: Telegram Bot Token belum diatur di file .env")
        return
        
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("critique", handle_critique))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    print("Bot sedang berjalan... Tekan Ctrl+C untuk berhenti.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
