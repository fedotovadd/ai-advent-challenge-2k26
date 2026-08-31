# Минимальный клиент DeepSeek

Программа принимает запрос в терминале, отправляет его в DeepSeek API и выводит ответ модели.

## 1. Получите API-ключ

Создайте ключ в [DeepSeek Platform](https://platform.deepseek.com/api_keys). Никому не отправляйте ключ и не записывайте его в исходный код.

## 2. Подготовьте окружение

Требуется Python 3.9 или новее.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Добавьте ключ

В macOS или Linux:

```bash
export DEEPSEEK_API_KEY="ваш_ключ"
```

## 4. Запустите программу

```bash
python3 main.py
```

Введите запрос и нажмите Enter. Ответ DeepSeek появится в консоли.
