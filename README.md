# mse_search_for_audio_replays

Задачи: https://github.com/moevm/mse_search_for_audio_replays/projects

## Инструкции по запуску

Основной скрипт -- src/main.py. Для получения справки, запустите его с
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
