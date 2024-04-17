import asyncio
import telegram_send
from telegram import Bot

# Import env variables
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("bot_token")

bot = Bot(token=token)

group_chat_id = '-4109659430'


async def send_telegram_message(title, message):

    msg = ""

    msg += f"{title}\n\n"

    msg += f"{message}"

    await bot.send_message(chat_id=group_chat_id, text=msg)

    pass


def send_message(title, message):
    asyncio.get_event_loop().run_until_complete(send_telegram_message(title, message))
    pass


if __name__ == "__main__":
    send_message("Test", "This is a test message")
    pass
