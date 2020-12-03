import os
import unittest

from src.config import Config
from src.utils.argument_parser import parse_args


class TestArgumentParser(unittest.TestCase):

    def setUp(self) -> None:

        self.config = Config()

    def test_something(self):

        path_file = os.path.join(self.config.path_data, "test", "test_flower.jpg")
        parsed_args = parse_args(args=["-f", path_file])

        self.assertIsInstance(parsed_args, dict)
        self.assertEqual(parsed_args["file_path"], path_file)


if __name__ == '__main__':
    unittest.main()
