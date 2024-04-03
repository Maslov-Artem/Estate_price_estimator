from aiogram import Router, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message, ReplyKeyboardRemove
from custom_filtrs.custom import ID_Filter
from funcs.cian_m import gimi_smth
from funcs.const import *

router = Router()


class CianState(StatesGroup):
    cian_id_true = State()

@router.message(StateFilter(None), F.text == catch_2)
async def cian_id(message: Message, state: FSMContext):
    await message.answer(
        text="Укажите ID:",
        reply_markup=None
    )
    await state.set_state(CianState.cian_id_true)


@router.message(CianState.cian_id_true, ID_Filter())
async def cian_answer(message: Message, state: FSMContext):
    await state.update_data(chosen_id=message.text)

    user_data = await state.get_data()

    id = user_data['chosen_id']

    res = requests.post(url_cian, data=id)

    res = json.loads(res.content)['classify_me']
    
    reply = f'Справедливая стоимость данного объекта: {res}₽'
    await message.answer(reply)

    await message.answer('/start')


    await state.clear()


################################################################
@router.message(CianState.cian_id_true, ~ID_Filter())
async def food_chosen(message: Message, state: FSMContext):

    await message.answer(f'''{message.text} кажется id нанастоящий...\nКинь ещё один или начни сначала: /start''')