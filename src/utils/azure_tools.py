"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 13/01/2021
"""

from azure.identity import AzureCliCredential
from azure.keyvault.secrets import SecretClient, KeyVaultSecret
from typing import Union

from src.utils.logger import Logger


class AzureKeyVault(object):

    def __init__(self, key_vault_name: str):

        credential = AzureCliCredential()
        key_vault_uri = f"https://{key_vault_name}.vault.azure.net"
        self.client = SecretClient(vault_url=key_vault_uri, credential=credential)

    def set_secret(self, secret_name: str, secret_value) -> bool:

        try:
            self.client.set_secret(secret_name, secret_value)
            return True
        except BaseException as e:
            Logger().error(f"{e}")
            return False

    def get_secret(self, secret_name: str) -> Union[KeyVaultSecret, str]:

        try:
            return self.client.get_secret(secret_name)
        except BaseException as e:
            Logger().error(f"{e}")
            return ""

    def delete_secret(self, secret_name: str) -> bool:

        try:
            poller = self.client.begin_delete_secret(secret_name)
            deleted_secret = poller.result()
            return True
        except BaseException as e:
            Logger().error(f"{e}")
            return False


def main():
    pass


if __name__ == "__main__":
    main()
