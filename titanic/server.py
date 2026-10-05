"""Local web UI and JSON prediction API. Run: python -m titanic.server."""

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit
import webbrowser

from .predict import ROOT, SurvivalPredictor

WEB_ROOT = ROOT / "web"
MAX_BODY = 8192


def reject_constant(value):
    raise ValueError(f"Invalid JSON number: {value}")


class TitanicServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        self.predictor = SurvivalPredictor()
        report = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))
        candidate = report["candidates"][report["selected_candidate"]]
        self.model_info = {
            "ready": True,
            "accuracy": report["test"]["accuracy"],
            "roc_auc": report["test"]["roc_auc"],
            "test_count": report["test"]["n"],
            "dataset_count": sum(split["rows"] for split in report["splits"].values()),
            "layers": candidate["layers"],
            "threshold": self.predictor.threshold,
        }
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "TitanicLocal/1.0"

    def send_bytes(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; "
                         "script-src 'self'; style-src 'self'; connect-src 'self'; "
                         "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def send_json(self, status, value):
        self.send_bytes(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                        "application/json; charset=utf-8")

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/api/model":
            self.send_json(200, self.server.model_info)
            return
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        if not target.is_relative_to(WEB_ROOT.resolve()) or not target.is_file():
            self.send_json(404, {"error": "Страница не найдена."})
            return
        if target.suffix not in {".html", ".css", ".js", ".png", ".webp", ".svg", ".woff2"}:
            self.send_json(404, {"error": "Файл не найден."})
            return
        content_type = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}.get(
            target.suffix, mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        if target.suffix in {".html", ".css", ".js", ".svg"}:
            content_type += "; charset=utf-8"
        self.send_bytes(200, target.read_bytes(), content_type)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/predict":
            self.send_json(404, {"error": "Метод не найден."})
            return
        origin = self.headers.get("Origin")
        port = self.server.server_port
        if origin and origin not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}:
            self.send_json(403, {"error": "Запрос разрешён только из локального приложения."})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.send_json(415, {"error": "Ожидаются данные в формате JSON."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BODY:
            self.send_json(413, {"error": "Некорректный размер запроса."})
            return
        try:
            self.connection.settimeout(5)
            passenger = json.loads(self.rfile.read(length).decode("utf-8"), parse_constant=reject_constant)
            result = self.server.predictor.predict(passenger)
        except (ValueError, UnicodeError) as exc:
            field = str(exc).split(":", 1)[0]
            fields = {"age": "возраст", "sex": "пол", "pclass": "класс билета", "fare": "стоимость билета",
                      "sibsp": "число братьев, сестёр и супругов", "parch": "число родителей и детей",
                      "deck": "палубу", "embarked": "порт посадки"}
            self.send_json(400, {"error": f"Проверьте {fields[field]}." if field in fields
                               else "Проверьте данные пассажира: запрос имеет неверный формат."})
            return
        except TimeoutError:
            self.send_json(408, {"error": "Время ожидания запроса истекло."})
            return
        self.send_json(200, result)


def main():
    parser = argparse.ArgumentParser(description="Titanic local game-style interface")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    try:
        server = TitanicServer(("127.0.0.1", args.port))
    except OSError as exc:
        parser.exit(1, f"Cannot start server: {exc}. Try --port 8001.\n")
    url = f"http://127.0.0.1:{args.port}"
    print(f"Titanic is ready: {url}\nPress Ctrl+C to stop.", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
