"""
========================================================================================
Handles connection to Azure, making it possible to read data from Azure file system.
========================================================================================
"""

from azure.identity import AzureCliCredential
from pyarrow.fs import PyFileSystem
import pyarrowfs_adlgen2

from azure.storage.filedatalake import DataLakeServiceClient


class FS(PyFileSystem):
    """
    This should take care of the file system
    """
    def __init__(self, storage_account) -> None:
        super().__init__(
            handler=pyarrowfs_adlgen2.AccountHandler.from_account_name(
                storage_account,
                AzureCliCredential()
            )
        )


class DataLake(list):
    def __init__(self, storage_account, path) -> None:

        self.base = path

        self.container = self.base.split("/")[0]

        self.service_client = DataLakeServiceClient(
            account_url="{}://{}.dfs.core.windows.net".format(
                "https",
                storage_account
            ),
            credential=AzureCliCredential()
        )

        self.fs = self.service_client.get_file_system_client(file_system=self.container)
        super().__init__(self.get_paths())

    def get_paths(self):

        paths = self.fs.get_paths(path="/".join(self.base.split("/")[1:]))

        parquets = []
        for path in paths:
            if ".parquet" in path.name and "delta_log" not in path.name:
                parquets.append(self.container + "/" + path.name)

        return parquets
