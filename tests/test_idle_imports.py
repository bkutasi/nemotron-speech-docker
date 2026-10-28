import subprocess
import sys
import unittest


class IdleImportsTest(unittest.TestCase):
    def test_startup_does_not_import_inference_libraries(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import app.main; "
                "assert not {'numpy', 'onnxruntime_genai', 'huggingface_hub'} & sys.modules.keys()",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
