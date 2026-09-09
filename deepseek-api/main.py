from providers import ask_model
from web import ChatServer


def create_server(host="127.0.0.1", port=8000, ask_model=ask_model, state_path=None):
    # Параметр ask_model позволяет тестам подменять провайдера без реального API-вызова.
    return ChatServer((host, port), ask_model, state_path)


def main():
    server = create_server()
    print("Откройте http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
