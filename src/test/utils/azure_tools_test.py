"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 13/01/2021
"""

import datetime
import unittest

from src.utils.azure_tools import AzureKeyVault
from src.utils.logger import Logger


class TestAzureTools(unittest.TestCase):
    def test_azure_key_vault(self):

        secret_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        key_vault = AzureKeyVault(key_vault_name="evolutionary-vault")
        key_vault.set_secret(secret_name=secret_name, secret_value="test")
        secret = key_vault.get_secret(secret_name=secret_name)
        self.assertEqual("test", secret.value)

        Logger().info(f"Deleting secret {secret_name}. Following ERROR message can be ignored")
        key_vault.delete_secret(secret_name)
        secret = key_vault.get_secret(secret_name=secret_name)
        self.assertEqual("", secret)


if __name__ == '__main__':
    unittest.main()
