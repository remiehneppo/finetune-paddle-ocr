import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ocr_labeler.cli import build_parser, build_settings, main, parse_args


class LabelerCLITests(unittest.TestCase):
    @staticmethod
    def make_model_dir(root: Path) -> Path:
        model_dir = root / "model"
        model_dir.mkdir()
        for name in (
            "inference.json",
            "inference.pdiparams",
            "inference.yml",
            "ppocr_keys.txt",
        ):
            (model_dir / name).touch()
        return model_dir

    def test_defaults_target_exported_model_and_single_gpu(self):
        settings = build_settings(build_parser().parse_args([]))

        self.assertEqual(settings.device, "gpu:0")
        self.assertEqual(settings.text_rec_input_shape, (3, 48, 1600))
        self.assertEqual(
            settings.rec_model_dir,
            Path("runs/vi_rec_3datasets_v1/inference/best_accuracy"),
        )
        self.assertIsNone(settings.det_model_dir)
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, 8010)

    def test_parse_args_accepts_explicit_cpu_and_model_directories(self):
        args = parse_args(
            [
                "--device",
                "cpu",
                "--rec-model-dir",
                "/models/rec",
                "--det-model-dir",
                "/models/det",
                "--host",
                "::1",
                "--port",
                "9000",
            ]
        )
        settings = build_settings(args)

        self.assertEqual(settings.device, "cpu")
        self.assertEqual(settings.rec_model_dir, Path("/models/rec"))
        self.assertEqual(settings.det_model_dir, Path("/models/det"))
        self.assertEqual(settings.host, "::1")
        self.assertEqual(settings.port, 9000)

    def test_main_opens_requested_workspace_and_forces_one_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            images = Path(directory)
            app = object()
            with (
                patch("ocr_labeler.cli.create_app", return_value=app) as create_app,
                patch("ocr_labeler.cli.uvicorn.run") as run,
            ):
                result = main(["--images", str(images), "--device", "cpu"])

        self.assertEqual(result, 0)
        settings = create_app.call_args.kwargs["settings"]
        self.assertEqual(settings.device, "cpu")
        self.assertEqual(
            create_app.call_args.kwargs["initial_workspace"],
            images,
        )
        run.assert_called_once_with(
            app,
            host="127.0.0.1",
            port=8010,
            workers=1,
        )

    def test_settings_reject_malformed_gpu_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = self.make_model_dir(Path(directory))
            for device in ("gpu:", "gpu:abc", "gpu:-1"):
                with self.subTest(device=device):
                    with self.assertRaisesRegex(
                        ValueError, r"device must be cpu or gpu:<index>"
                    ):
                        build_settings(
                            parse_args(
                                [
                                    "--rec-model-dir",
                                    str(model_dir),
                                    "--device",
                                    device,
                                ]
                            )
                        ).validate()

    def test_settings_reject_ports_outside_tcp_range(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = self.make_model_dir(Path(directory))
            for port in (0, 65536):
                with self.subTest(port=port):
                    with self.assertRaisesRegex(
                        ValueError, r"port must be between 1 and 65535"
                    ):
                        build_settings(
                            parse_args(
                                [
                                    "--rec-model-dir",
                                    str(model_dir),
                                    "--port",
                                    str(port),
                                ]
                            )
                        ).validate()

    def test_settings_accept_gpu_zero_and_valid_port_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = self.make_model_dir(Path(directory))
            for port in (1, 65535):
                with self.subTest(port=port):
                    settings = build_settings(
                        parse_args(
                            [
                                "--rec-model-dir",
                                str(model_dir),
                                "--device",
                                "gpu:0",
                                "--port",
                                str(port),
                            ]
                        )
                    )
                    self.assertIs(settings.validate(), settings)

    def test_settings_accept_only_loopback_hosts(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = self.make_model_dir(Path(directory))
            for host in ("localhost", "127.0.0.1", "127.23.45.67", "::1"):
                with self.subTest(host=host):
                    settings = build_settings(
                        parse_args(
                            [
                                "--rec-model-dir",
                                str(model_dir),
                                "--host",
                                host,
                            ]
                        )
                    )
                    self.assertIs(settings.validate(), settings)

    def test_settings_reject_non_loopback_hosts(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = self.make_model_dir(Path(directory))
            for host in ("0.0.0.0", "::", "192.168.1.5", "example.com"):
                with self.subTest(host=host):
                    with self.assertRaisesRegex(
                        ValueError, "host must be localhost or a loopback address"
                    ):
                        build_settings(
                            parse_args(
                                [
                                    "--rec-model-dir",
                                    str(model_dir),
                                    "--host",
                                    host,
                                ]
                            )
                        ).validate()


if __name__ == "__main__":
    unittest.main()
