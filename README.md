# ai-advent-challenge-2k26
AI Advent Challenge by Alexey Gladkov

## Задания

- [День 1 — минимальный CLI-клиент DeepSeek](day-01-deepseek-api/)

## Структура репозитория

```text
.
├── README.md
├── .gitignore
└── day-01-deepseek-api/
    ├── README.md
    ├── main.py
    ├── requirements.txt
    ├── tests/
    │   └── test_main.py
    └── docs/
        └── superpowers/
            ├── plans/
            │   └── 2026-08-31-deepseek-cli.md
            └── specs/
                └── 2026-08-31-deepseek-cli-design.md
```

- `day-01-deepseek-api/main.py` — минимальный CLI-клиент для отправки запроса в DeepSeek.
- `day-01-deepseek-api/requirements.txt` — список Python-зависимостей.
- `day-01-deepseek-api/tests/` — автоматические тесты программы.
- `day-01-deepseek-api/docs/` — дизайн и план реализации первого задания.
- `.gitignore` — исключения для виртуального окружения, Python-кэша и локальных секретов.
