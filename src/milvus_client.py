import logging
import requests
import json

from pymilvus import (
    connections,
    MilvusException,
    utility,
    Prepare,
    Collection,
    CollectionSchema,
    FieldSchema,
)
from pymilvus.decorators import retry_on_rpc_failure


class ConnectParams:
    def __init__(
        self,
        alias: str = "default",
        host: str = "localhost",
        port: int = 19530,
        username: str = "",
        password: str = "",
        database: str = "default",
    ):
        self.alias = alias
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database


class Milvus:
    def __init__(self, connect_params: ConnectParams):
        self._connect_params = connect_params
        self._using = connect_params.alias
        self._connect(
            connect_params.alias,
            connect_params.host,
            connect_params.port,
            connect_params.username,
            connect_params.password,
            connect_params.database,
        )

    def _connect(
        self,
        alias: str = "default",
        host: str = "localhost",
        port: int = 19530,
        username: str = None,
        password: str = None,
        database: str = "default",
    ):
        try:
            logging.debug(
                f"Connecting to Milvus http://{username}:{password}@{host}:{port}/{database}..."
            )
            connections.connect(
                alias=alias,
                host=host,
                port=port,
                db_name=database,
                token=f"{username}:{password}",
            )
        except MilvusException:
            logging.error("Milvus connection failed")

    def _get_connection(self):
        return connections._fetch_handler(self._using)

    def has_collection(self, collection_name: str):
        return utility.has_collection(collection_name, using=self._using)

    def has_partition(self, collection_name: str, partition_name: str):
        return utility.has_partition(collection_name, partition_name, using=self._using)

    def describe_collection(self, collection_name: str):
        conn = self._get_connection()
        return conn.describe_collection(collection_name)

    def describe_index(self, collection_name: str, index_name: str):
        conn = self._get_connection()
        return conn.describe_index(collection_name, index_name)

    def list_indexes(self, collection_name: str):
        conn = self._get_connection()
        return conn.list_indexes(collection_name)

    def list_partitions(self, collection_name: str):
        conn = self._get_connection()
        return conn.list_partitions(collection_name)

    @retry_on_rpc_failure()
    def _get_partitions_info(self, collection_name):
        conn = self._get_connection()
        req = Prepare.show_partitions_request(collection_name)
        rf = conn._stub.ShowPartitions.future(req, timeout=None)
        response = rf.result()
        return response

    def get_partition_id(self, collection_name, partition_name):
        response = self._get_partitions_info(collection_name)
        partition_ids = list(response.partitionIDs)
        partition_names = list(response.partition_names)
        assert len(partition_ids) == len(
            partition_names
        ), "[MilvusStorageClient] partition ids and partition names does not match!"
        idx = partition_names.index(partition_name)
        return partition_ids[idx]

    def create_collection(
        self, collection_name: str, schema: CollectionSchema, alias: str = ""
    ):
        if self.has_collection(collection_name):
            logging.info(f"Collection {collection_name} already exists")
            return

        Collection(name=collection_name, schema=schema, using=self._using, shards_num=2)
        if alias:
            self.create_alias(collection_name, alias)

    def create_alias(self, collection_name: str, alias: str):
        self._get_connection().create_alias(collection_name, alias)

    def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_type: str = "",
        metric_type: str = "",
        index_name: str = "",
    ):
        collection = Collection(collection_name, using=self._using)
        index_params = {}
        if index_type:
            index_params["index_type"] = index_type
        if metric_type:
            index_params["metric_type"] = metric_type
        collection.create_index(
            field_name, index_params=index_params, index_name=index_name
        )

    def load_collection(self, collection_name: str):
        Collection(collection_name, using=self._using).load(replica_number=2)

    def import_data(self, collection_name, files: list[str]) -> str:
        if not files:
            logging.warning("No files to import")
            return

        uri = f"{self._connect_params.host}:{self._connect_params.port}"
        url = f"http://{uri}/v2/vectordb/jobs/import/create"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._connect_params.username}:{self._connect_params.password}",
        }

        data = {
            "files": [[file] for file in files],
            "collectionName": collection_name,
            "dbName": self._connect_params.database,
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        logging.info(f"request: {json.dumps(data)}")
        logging.info(f"response: {response.text}")
        # validate response
        if (
            response.status_code != 200
            or "code" not in response.json()
            or (response.json()["code"] not in [0, 200])
        ):
            logging.error(f"Failed to import data: {response.json()}")
            raise Exception(f"Failed to import data, {response.json()}")

        return response.json()["data"]["jobId"]

    def get_import_job(self, job_id: str):
        uri = f"{self._connect_params.host}:{self._connect_params.port}"
        url = f"http://{uri}/v2/vectordb/jobs/import/get_progress"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._connect_params.username}:{self._connect_params.password}",
        }

        data = {"jobId": job_id}
        response = requests.post(url, headers=headers, data=json.dumps(data))

        return response.json()

    def list_import_job(self, collection_name):
        uri = f"{self._connect_params.host}:{self._connect_params.port}"
        url = f"http://{uri}/v2/vectordb/jobs/import/list"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._connect_params.username}:{self._connect_params.password}",
        }

        data = {
            "collectionName": collection_name,
            "dbName": self._connect_params.database,
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))

        return response.json()

    def parse_schema(self, collection_name: str) -> CollectionSchema:
        collection_info = self.describe_collection(collection_name)

        fields = []
        for field_info in collection_info["fields"]:
            fields.append(
                FieldSchema(
                    name=field_info["name"],
                    dtype=field_info["type"],
                    description=field_info["description"],
                    # check field_info["params"] has key
                    is_primary=(
                        field_info["is_primary"]
                        if "is_primary" in field_info
                        else False
                    ),
                    auto_id=(
                        field_info["auto_id"] if "auto_id" in field_info else False
                    ),
                    **field_info["params"],
                )
            )

        return CollectionSchema(
            fields=fields,
            description=collection_info["description"],
        )


def create_milvus_client(connect_params: list[ConnectParams]):
    return [Milvus(params) for params in connect_params]
