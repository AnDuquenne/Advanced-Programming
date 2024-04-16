import asyncio
import telegram_send


async def send_telegram_message(title, message):

    msg = ""

    msg += f"{title}\n\n"

    msg += f"{message}"

    await telegram_send.send(messages=[msg])

    pass


def send_message(title, message):
    asyncio.get_event_loop().run_until_complete(send_telegram_message(title, message))
    pass