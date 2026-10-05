"""Entry point for the standalone Windows executable."""

import argparse
import json
import sys
import threading
import urllib.request
import webbrowser

from titanic.server import TitanicServer


def smoke_test():
    """Exercise the bundled model, API, and assets without opening a browser."""
    server = TitanicServer(("127.0.0.1", 0))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    # A local application must not send localhost requests through a proxy.
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for resource in ["/", "/styles.css", "/app.js", "/favicon.svg", "/assets/titanic-night.png"]:
            with client.open(url + resource, timeout=10) as response:
                if response.status != 200 or not response.read():
                    raise RuntimeError(f"Missing bundled asset: {resource}")
        passenger = {"sex": "female", "pclass": 2, "age": 28, "sibsp": 0, "parch": 0,
                     "fare": None, "deck": "UNKNOWN", "embarked": "S"}
        request = urllib.request.Request(url + "/api/predict", data=json.dumps(passenger).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with client.open(request, timeout=10) as response:
            result = json.load(response)
        expected = 0.7712102788360875
        if abs(result["survival_probability"] - expected) > 1e-10:
            raise RuntimeError("Bundled model prediction differs from the verified model")
        with client.open(url + "/api/model", timeout=10) as response:
            model = json.load(response)
        if model["dataset_count"] != 1309 or model["test_count"] != 261:
            raise RuntimeError("Bundled model metadata is invalid")
        print(json.dumps({"status": "ok", "frozen": bool(getattr(sys, "frozen", False)),
                          "bundled_assets": 5, "prediction": result}, ensure_ascii=False), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def main():
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Titanic — ваша история на борту")
    parser.add_argument("--port", type=int, default=0, help="Port; default: automatically choose a free port")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open a browser")
    parser.add_argument("--smoke-test", action="store_true", help="Verify bundled model and UI, then exit")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535")
    if args.smoke_test:
        smoke_test()
        return
    server = TitanicServer(("127.0.0.1", args.port))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"\n  TITANIC — ваша история на борту\n\n  Приложение готово: {url}\n"
          "\n  Оставьте это окно открытым, пока пользуетесь приложением.\n"
          "  Для выхода закройте это окно или нажмите Ctrl+C.\n", flush=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        if not args.no_browser:
            if not webbrowser.open(url):
                print("  Не удалось открыть браузер автоматически. Откройте адрес выше.", flush=True)
        while worker.is_alive():
            worker.join(timeout=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nНе удалось запустить Titanic: {exc}", file=sys.stderr, flush=True)
        if sys.stdin is not None and sys.stdin.isatty():
            input("Нажмите Enter, чтобы закрыть окно…")
        raise SystemExit(1)
