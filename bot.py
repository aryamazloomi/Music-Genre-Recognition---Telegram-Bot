#importing libraries

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt

import os
from PIL import Image
from pathlib import Path
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import tensorflow
import librosa.display
import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import backend, Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, GRU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D

from tensorflow.keras import regularizers

model = keras.models.load_model('')

def getGenre(file_download_path):
    songname = file_download_path
    genres = ['Blues', 'Classical', 'Country', 'Disco', 'HipHop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'] #get labels
    for index in range(14):
        y, sr = librosa.load(songname, mono=True, duration=2, offset=index*2)
        ps = librosa.feature.melspectrogram(y=y, sr=sr, hop_length = 256, n_fft = 512, n_mels=64)
        ps = librosa.power_to_db(ps**2)
    tst = np.array([ps.reshape( (64, 173,1))]) #get audio features
    tst.shape
    result = list(model.predict(tst))
    result = result[0]
    max_val = max(result)
    result = result.tolist()
    print("---------------------")
    print(result)

    gen = genres[result.index(max_val)]
    print(gen)
    print("Sucessful")
    return gen
############################ telegram bot starts from here #######################################

import logging

from telegram import Update, ForceReply, ChatAction
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\! ',
       # reply_markup=ForceReply(selective=True),
    )
    update.message.reply_text('You can send me your music file and I\'ll detect its genre using my neural network!  Remember, it should be more than 30 seconds')


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('You can send me your music file and I\'ll detect its genre using my neural network! Remember, it should be more than 30 seconds')


def echo(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("Please send an audio file.")







def download_file(user_id: int, file_to_download, file_type: str, context: CallbackContext) -> str:
    """Download a file using convenience methods of "python-telegram-bot"
    **Keyword arguments:**
     - user_id (int) -- The user's id
     - file_to_download (*) -- The file object to download
     - file_type (str) -- The type of the file, either 'photo' or 'audio'
     - context (CallbackContext) -- The context object of the user
    **Returns:**
     The path of the downloaded file
    """
    user_download_dir = f"downloads/{user_id}"
    file_id = ''
    file_extension = ''

    if file_type == 'audio':
        file_id = context.bot.get_file(file_to_download.file_id)
        file_name = file_to_download.file_name
        file_extension = file_name.split(".")[-1]
    elif file_type == 'voice':
        file_id = 'voice'
        file_extension = 'ogg'

    file_download_path = f"{file_id.file_id}.{file_extension}"



    try:
        file_id.download(f"{file_id.file_id}.{file_extension}")
    except ValueError as error:
        raise Exception(f"Couldn't download the file with file_id: {file_id}") from error

    return file_download_path

def download_voice(user_id: int, file_to_download, file_type: str, context: CallbackContext) -> str:
    """Download a file using convenience methods of "python-telegram-bot"
    **Keyword arguments:**
     - user_id (int) -- The user's id
     - file_to_download (*) -- The file object to download
     - file_type (str) -- The type of the file, either 'photo' or 'audio'
     - context (CallbackContext) -- The context object of the user
    **Returns:**
     The path of the downloaded file
    """


    file_id = context.bot.get_file(file_to_download.file_id)
    #file_name = file_to_download.file_name
    file_extension = 'oga'

    file_download_path = f"tmp.oga"

    try:
        file_id.download(f"tmp.oga")
    except ValueError as error:
        raise Exception(f"Couldn't download the file with file_id: {file_id}") from error

    return file_download_path




def handle_music_message(update: Update, context: CallbackContext) -> None:
    message = update.message
    user_id = update.effective_user.id
    music_duration = message.audio.duration
    music_file_size = message.audio.file_size
    print(music_duration)

    if music_duration >= 3600 and music_file_size > 20000000:
        message.reply_text(
            'file is too large')

        return


    context.bot.send_chat_action(
        chat_id=message.chat_id,
        action=ChatAction.TYPING
    )

    try:
        file_download_path = download_file(
            user_id=user_id,
            file_to_download=message.audio,
            file_type='audio',
            context=context
        )
    except ValueError:
        message.reply_text(
        )
        logger.error("Error on downloading %s's file. File type: Audio", user_id, exc_info=True)
        return
    gnr = getGenre(file_download_path)
    update.message.reply_text(gnr)
    if os.path.exists(file_download_path):
        os.remove(file_download_path)




def handle_voice_message(update: Update, context: CallbackContext) -> None:
    message = update.message
    user_id = update.effective_user.id
    music_duration = message.voice.duration
    music_file_size = message.voice.file_size
    print(music_duration)

    if music_duration >= 3600 and music_file_size > 48000000:
        message.reply_text(
            'file is too large')

        return


    context.bot.send_chat_action(
        chat_id=message.chat_id,
        action=ChatAction.TYPING
    )

    try:
        file_download_path = download_voice(
            user_id=user_id,
            file_to_download=message.voice,
            file_type='audio',
            context=context
        )
    except ValueError:
        message.reply_text(
        )
        logger.error("Error on downloading %s's file. File type: Audio", user_id, exc_info=True)
        return
    gnr = getGenre(file_download_path)
    update.message.reply_text(gnr)
    if os.path.exists(file_download_path):
        os.remove(file_download_path)










def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater("Your Token")

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.audio, handle_music_message))
    dispatcher.add_handler(MessageHandler(Filters.voice, handle_voice_message))

    # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()





