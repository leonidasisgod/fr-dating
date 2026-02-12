import os
import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from openai import OpenAI

# Твои модули
from src.load_data import load_profiles
from src.profile_text import profile_to_text
from src.embeddings import embed_texts, normalize_embeddings
from src.retrieval import VectorIndex

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
dp = Dispatcher()

# Инициализация движка
profiles = load_profiles("data/raw_profiles.json")
texts = [profile_to_text(p) for p in profiles]
embs = normalize_embeddings(embed_texts(texts))
index = VectorIndex(dim=embs.shape[1])
index.add(embs, profiles)


async def get_ai_verdict(user, cand, mode="match"):
    if mode == "couple":
        prompt = f"Analyze relationship compatibility: {user.bio} AND {cand.bio}. 3 sentences. Be deep, poetic, and slightly romantic. Mention their shared energy."
    else:
        prompt = f"Analyze synergy: {user.bio} vs {cand.bio}. 2 witty sentences, no direct quotes."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    builder = InlineKeyboardBuilder()
    # Кнопки выбора персонажа
    for p in profiles[:4]:
        builder.button(text=f"🎭 Play as {p.id}", callback_data=f"login_{p.id}")

    # ТА САМАЯ КНОПКА ДЛЯ ВАС
    builder.button(text="💖 НАША СОВМЕСТИМОСТЬ (u101 + u102)", callback_data="check_couple")
    builder.adjust(2)

    await message.answer(
        "✨ **Neural Matchmaker v3.0** ✨\n\nДобро пожаловать в игру! Выбери персонажа или проверь вашу связь:",
        reply_markup=builder.as_markup()
    )


@dp.callback_query(F.data == "check_couple")
async def handle_couple_check(callback: types.CallbackQuery):
    try:
        user_a = next(p for p in profiles if p.id == "u101")
        user_b = next(p for p in profiles if p.id == "u102")
    except StopIteration:
        await callback.message.answer("❌ Ошибка: Профили u101 и u102 не найдены в JSON!")
        return

    await callback.message.answer("🔮 **ИИ начинает глубокий анализ вашей химии...**")
    verdict = await get_ai_verdict(user_a, user_b, mode="couple")

    await callback.message.answer(
        f"👩‍❤️‍👨 **АНАЛИЗ ПАРЫ: {user_a.id} + {user_b.id}**\n\n"
        f"{verdict}",
        parse_mode="Markdown"
    )


@dp.callback_query(F.data.startswith("login_"))
async def handle_login(callback: types.CallbackQuery):
    user_id = callback.data.split("_")[1]
    user = next(p for p in profiles if p.id == user_id)

    me_emb = embed_texts([profile_to_text(user)])
    matches = index.search_hybrid(user, me_emb, k=2, alpha=0.7)

    await callback.message.answer(f"✅ Ты в роли **{user.id}**. Ищем твою судьбу...")

    for m in matches:
        cand = m['profile']
        text = f"🔥 **МЭТЧ: {int(m['score'] * 100)}%**\n👤 {cand.id} | {cand.location}\n📝 Bio: {cand.bio}"
        msg = await callback.message.answer(text)

        verdict = await get_ai_verdict(user, cand)
        await msg.edit_text(text + f"\n\n🤖 **AI VERDICT:**\n_{verdict}_", parse_mode="Markdown")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())