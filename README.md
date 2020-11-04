# mse_search_for_audio_replays

Задачи: https://github.com/moevm/mse_search_for_audio_replays/projects

Вся основная информация прописана на wiki: https://github.com/moevm/mse_search_for_audio_replays/wiki

# Инструкция по установке

Для установки зависимостей используйте файл requirements.txt:
```sh
pip3 install -r requirements.txt
```

Для работы пакета pydub нужна программа ffmpeg. Как правило, её можно
получить в одноимённом пакете, например на Ubuntu:
```sh
sudo apt install ffmpeg
```

# Инструкция по запуску

Основной скрипт — src/main.py. Для получения справки, запустите его с
ключом `--help`:
```sh
src/main.py --help
```

У программы есть два режима работы: фильтрация шумов и обнаружение
повторов. По-умолчанию выбран режим обнаружения повторов. Чтобы выбрать
режим фильтрации шумов, нужно указать флаг `--denoise` и передать
образец шумов с флагом `--noise-sample`.

Пока что вместо обоих режимов работы поставлены заглушки, которые
выводят режим и набор переданных имён файлов.

Архив, содержащий видео с примерами работы интерфейса программы: [скачать ZIP](https://github.com/moevm/mse_search_for_audio_replays/files/5323273/screen_cast.zip)
