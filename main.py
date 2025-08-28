"""
Telegram Audio Transcription Bot
--------------------------------

This script implements a simple Telegram bot capable of transcribing
almost any audio file into text.  It downloads audio sent by users,
transcodes it into a format accepted by the OpenAI Whisper API and
returns the resulting transcript.

Features:

* Supports voice messages (Telegram sends these as OGG/Opus) and
  general audio files (e.g. MP3, WAV, M4A, etc.).
* Uses the OpenAI Whisper cloud API by default for high‑quality
  transcription.  Local transcription could also be plugged in by
  replacing the ``transcribe_audio`` function.
* Handles arbitrary file sizes by streaming the file in chunks and
  storing it temporarily on disk.
* Written for ``python‑telegram‑bot`` version 20 and later.  Earlier
  versions use a different API; see the library documentation for
  details.

Prerequisites:

* Python 3.9+
* ``python‑telegram‑bot`` (install via ``pip install python‑telegram‑bot``)
* ``pydub`` for audio transcoding (``pip install pydub``)
* ``ffmpeg`` binary available in your ``PATH``; on Debian/Ubuntu
  ``sudo apt install ffmpeg``.  ``pydub`` calls ffmpeg behind the
  scenes to convert between formats.  Without ffmpeg installed the
  bot will not be able to handle certain audio types.
* A Telegram bot token – create one via ``@BotFather`` and set it as
  the environment variable ``TELEGRAM_BOT_TOKEN``.
* An OpenAI API key for the Whisper transcription endpoint – set it
  as the environment variable ``OPENAI_API_KEY``.  See  https://platform.openai.com/docs/api-reference/audio/create
  for details.

Usage:

1. Install dependencies and ensure ffmpeg is available.
2. Export environment variables:

   ``export TELEGRAM_BOT_TOKEN="<your bot token>"``
   ``export OPENAI_API_KEY="<your OpenAI API key>"``

3. Run the bot:

   ``python telegram_audio_transcriber_bot.py``

4. Send an audio file or voice message to your bot in Telegram.  The
   bot will respond with the transcribed text.

Note:

This bot uses the OpenAI Whisper API by default, which incurs cost
according to OpenAI’s pricing.  Replace the ``transcribe_audio``
function with a local transcription implementation (e.g. using
``openai-whisper`` or ``vosk``) if you prefer not to rely on external
services.

"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from pydub import AudioSegment


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def ensure_ffmpeg() -> None:
    """Ensure that ffmpeg is available on the system.

    Pydub relies on ffmpeg for format conversions.  This helper will
    check whether ffmpeg is in the PATH and raise an informative
    message otherwise.
    """
    from shutil import which
    if which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg was not found on your system. Please install ffmpeg and make sure it "
            "is available in your PATH. On Debian/Ubuntu run 'sudo apt install ffmpeg'."
        )


def transcribe_audio(file_path: Path, language: str = "auto") -> str:
    """Transcribe an audio file using the OpenAI Whisper API.

    :param file_path: Path to the audio file to transcribe.  The file
                      must be readable in binary mode.
    :param language:  BCP‑47 language code (e.g. "uk", "en"), or "auto"
                      to let Whisper detect the language automatically.
    :returns:        The transcribed text.
    :raises ValueError: if the OpenAI API key is missing or the API
                        request fails.

    This function posts the audio to OpenAI's ``/v1/audio/transcriptions``
    endpoint.  It expects the environment variable ``OPENAI_API_KEY``
    to be set with a valid key.  See https://platform.openai.com/docs/api-reference/audio
    for more details.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")

    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    with open(file_path, "rb") as audio_file:
        files = {"file": (file_path.name, audio_file, "application/octet-stream")}
        data = {
            "model": "whisper-1",
        }
        if language != "auto":
            data["language"] = language
        logger.info("Sending audio for transcription…")
        response = requests.post(url, headers=headers, files=files, data=data)
    if response.status_code != 200:
        logger.error("Failed transcription: %s", response.text)
        raise ValueError(f"OpenAI API returned an error: {response.text}")
    result = response.json()
    return result.get("text", "")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "Привіт! Надішліть аудіофайл або голосове повідомлення, і я надішлю вам його "
        "текстову версію.\n\n"
        "Для коректної роботи потрібно заздалегідь вказати ключ API OpenAI (перемінна "
        "OPENAI_API_KEY) та встановити ffmpeg."
    )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming audio files and voice messages."""
    ensure_ffmpeg()
    message = update.effective_message
    if not message:
        return

    # Telegram uses different fields for voice vs audio messages
    tg_file = message.voice or message.audio
    if tg_file is None:
        await message.reply_text("Надішліть, будь ласка, аудіофайл або голосове повідомлення.")
        return

    # Download the file to a temporary location
    file = await context.bot.get_file(tg_file.file_id)
    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = Path(tmpdir) / tg_file.file_name if tg_file.file_name else Path(tmpdir) / "input"
        await file.download_to_drive(custom_path=str(original_path))
        logger.info("Downloaded file to %s", original_path)

        # Convert the file to a format accepted by Whisper API (mp3)
        # Some formats (e.g. OGG/Opus) are not accepted directly; mp3 is a safe bet.
        converted_path = Path(tmpdir) / (original_path.stem + ".mp3")
        try:
            audio = AudioSegment.from_file(original_path)
            audio.export(converted_path, format="mp3")
            logger.info("Converted %s to %s", original_path, converted_path)
        except Exception as exc:
            logger.exception("Failed to convert audio file: %s", exc)
            await message.reply_text(
                "Не вдалося обробити аудіофайл. Перевірте формат і спробуйте ще раз."
            )
            return

        # Transcribe the converted audio
        try:
            transcript = transcribe_audio(converted_path)
        except Exception as exc:
            logger.exception("Error during transcription: %s", exc)
            await message.reply_text(
                "Виникла помилка під час розпізнавання. Переконайтеся, що API ключ правильний "
                "та сервер OpenAI доступний."
            )
            return

        if not transcript:
            await message.reply_text("Не вдалося розпізнати мову або отримати текст.")
        else:
            # Send the transcript back to the user
            await message.reply_text(transcript)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "Цей бот перетворює аудіофайли на текст. Надішліть аудіо або голосове, "
        "і через деякий час ви отримаєте текстову версію. Підтримуються різні "
        "формати аудіо, але потрібно встановити ffmpeg."
    )


def main() -> None:
    """Start the bot."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable must be set")
    application = ApplicationBuilder().token(bot_token).build()

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Message handlers
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

    # Start the bot
    logger.info("Starting Telegram audio transcriber bot…")
    application.run_polling()


if __name__ == "__main__":
    main()
