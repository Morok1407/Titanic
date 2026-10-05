import http.client
import json
import threading
import unittest

from titanic.server import TitanicServer


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = TitanicServer(("127.0.0.1", 0))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def predict(self, passenger):
        status, _, body = self.request("POST", "/api/predict", json.dumps(passenger), {"Content-Type": "application/json"})
        return status, json.loads(body)

    def test_web_prediction_matches_saved_neural_network(self):
        passenger = {"sex": "female", "pclass": 2, "age": 28, "sibsp": 0, "parch": 0,
                     "fare": None, "deck": "UNKNOWN", "embarked": "S"}
        status, actual = self.predict(passenger)
        self.assertEqual(status, 200)
        self.assertEqual(actual, self.server.predictor.predict(passenger))

    def test_both_outcomes_and_unknown_fields(self):
        for passenger in [{"sex": "male", "pclass": 3, "age": 30},
                          {"sex": "female", "pclass": 1, "age": 30},
                          {"sex": "female", "pclass": 2, "age": None, "fare": None}]:
            with self.subTest(passenger=passenger):
                status, result = self.predict(passenger)
                self.assertEqual(status, 200)
                self.assertTrue(0 <= result["survival_probability"] <= 1)
        self.assertEqual(self.predict({"sex": "male", "pclass": 3, "age": 30})[1]["predicted_survived"], 0)
        self.assertEqual(self.predict({"sex": "female", "pclass": 1, "age": 30})[1]["predicted_survived"], 1)

    def test_invalid_data_is_rejected_without_crashing(self):
        for value in [{"sex": "male", "pclass": 8}, {"sex": "female", "pclass": 1, "age": -10},
                      {"sex": "female", "pclass": 1, "age": float("nan")},
                      {"sex": "female", "pclass": 1, "boat": "1"}, [], None]:
            with self.subTest(value=value):
                status, result = self.predict(value)
                self.assertEqual(status, 400)
                self.assertIn("error", result)
        status, _, _ = self.request("POST", "/api/predict", "{bad", {"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_model_info_matches_report(self):
        status, _, body = self.request("GET", "/api/model")
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertTrue(result["ready"])
        self.assertEqual(result["dataset_count"], 1309)
        self.assertEqual(result["test_count"], 261)

    def test_static_assets_and_no_project_file_access(self):
        for path, expected_type in [("/", "text/html"), ("/styles.css", "text/css"),
                                    ("/app.js", "text/javascript"), ("/assets/titanic-night.png", "image/png")]:
            with self.subTest(path=path):
                status, headers, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertTrue(headers["Content-Type"].startswith(expected_type))
                self.assertGreater(len(body), 0)
        for path in ["/../data/titanic.csv", "/%2e%2e/models/titanic_mlp.npz", "/.env", "/missing"]:
            self.assertEqual(self.request("GET", path)[0], 404)

    def test_request_boundaries(self):
        self.assertEqual(self.request("POST", "/api/predict", "{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/predict", "{}",
                         {"Content-Type": "application/json", "Content-Length": "9000"})[0], 413)
        self.assertEqual(self.request("POST", "/api/predict", "{}",
                         {"Content-Type": "application/json", "Origin": "https://example.com"})[0], 403)


if __name__ == "__main__":
    unittest.main()
