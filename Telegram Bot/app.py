import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from rag_chain import generate_answer
from vision import describe_image
from memory import add_to_memory, get_memory

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot Ready! Use /help")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/ask <query>\n/image (send image)\n/help"
    )

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.chat_id
    query = " ".join(context.args)

    if not query:
        await update.message.reply_text("Enter a question")
        return

    history = get_memory(user_id)
    answer, sources = generate_answer(query, history)

    add_to_memory(user_id, query, answer)

    response = f" {answer}\n\n {', '.join(sources)}"

    await update.message.reply_text(response)

async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()

    path = "temp.jpg"
    await file.download_to_drive(path)

    caption, tags = describe_image(path)
    os.remove(path)

    await update.message.reply_text(
        f" {caption}\n {', '.join(tags)}"
    )

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(MessageHandler(filters.PHOTO, image_handler))

    print("Bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()