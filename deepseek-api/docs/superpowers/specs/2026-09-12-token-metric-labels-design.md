# Plain-language token metrics

## Goal

Make the metrics panel understandable without API terminology while preserving the existing token values and light visual style.

## Layout

The last-step section has five cards. Four cards remain in a two-column grid; the final context card spans both columns.

| Position | Label | Value |
| --- | --- | --- |
| 1 | Токенов в сообщении | Approximate tokens in the latest user message. |
| 2 | Токенов на входе | Tokens sent to the model: history, system prompt, and message. |
| 3 | Токенов в ответе | Tokens in the latest model response. |
| 4 | Вся история (накоплено) | Approximate tokens in the conversation after the latest response. |
| 5 | Контекст / лимит | Current request tokens and the model context limit. |

The card for history before the request is removed. The session summary starts with «Сообщений» instead of «Реплик».

## Implementation boundaries

Only `static/index.html` and its static-page tests change. The existing metric fields, API, persistence, calculations, graph, and history table stay intact.

## Verification

The static-page test checks the new labels, absence of «История до», and the full-width context-card hook. The full Python test suite must remain green.
