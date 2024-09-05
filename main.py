from collections import defaultdict
import glob
import io
import os
import logging
import shutil
import struct

import pandas as pd
from pymilvus import DataType, CollectionSchema
from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType
from milvus_client import ConnectParams, create_milvus_client
from oss_storage import OssConnectParam, create_oss_storage
from google.protobuf.json_format import MessageToDict

logging.getLogger("local_bulk_writer").setLevel(logging.ERROR)
logging.getLogger("oss2").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)


class MilvusMigrationTool:
    def __init__(
        self,
        source_milvus: ConnectParams,
        destination_milvus: ConnectParams,
        source_oss: OssConnectParam,
        destination_oss: OssConnectParam,
    ):
        self._src_client, self._dest_client = create_milvus_client(
            [source_milvus, destination_milvus]
        )
        self._src_bucket, self._dest_bucket = create_oss_storage(
            [source_oss, destination_oss]
        )

    def _migration_schema(self, collection: str):
        schema = self._src_client.parse_schema(collection)
        self._dest_client.create_collection(collection_name=collection, schema=schema)

    def _migration_index(self, collection: str):
        indexes_msg = self._src_client.list_indexes(collection)

        for msg in indexes_msg:
            d = MessageToDict(msg)
            index_name = d["indexName"]
            if not self._dest_client.describe_index(collection, index_name):
                logging.info(
                    f"Creating index, collection: {collection}, index: {index_name}"
                )
                index_info = self._src_client.describe_index(collection, index_name)
                self._dest_client.create_index(
                    collection,
                    field_name=index_info["field_name"],
                    index_type=index_info["index_type"],
                    metric_type=index_info["metric_type"],
                    index_name=index_name,
                )

    def _parse_storage_data(self, data, field_info):
        insert_offset = struct.unpack("<i", data[17:21])[0]
        payload_type = struct.unpack("<i", data[69:73])[0]
        payload_offset = insert_offset + 33
        payload_end = struct.unpack(
            "<i", data[insert_offset + 13 : insert_offset + 17]
        )[0]
        payload_length = payload_end - payload_offset
        payload = struct.unpack(
            f"<{payload_length}s", data[payload_offset:payload_end]
        )[0]
        pq_file = io.BytesIO(payload)
        df = pd.read_parquet(pq_file)

        res = []
        assert (
            payload_type == field_info["type"]
        ), f'[MilvusStorageClient] Field type does not match! {payload_type} != {field_info["type"]}'
        if payload_type == DataType.FLOAT_VECTOR.value:
            dim_num = field_info["params"]["dim"]
            for _, row in df.iterrows():
                val = row["val"]
                feature = struct.unpack(f"<{dim_num}f", val)
                res.append(feature)
        else:
            res = df["val"].tolist()
        return res

    def _get_storage_data(self, path, field_infos) -> dict:
        partition_data = defaultdict(list)
        for bin_seg_path in self._src_bucket.list_objects(path):
            for field_info in field_infos:
                field_id = str(field_info["field_id"])
                field_name = field_info["name"]
                bin_seg_field_path = os.path.join(bin_seg_path, field_id)
                for bin_seg_field_log_path in self._src_bucket.list_objects(
                    bin_seg_field_path
                ):
                    bytes_data = self._src_bucket.download_object(
                        bin_seg_field_log_path
                    )
                    data = self._parse_storage_data(bytes_data, field_info)
                    partition_data[field_name].extend(data)
        return partition_data

    def _read_storage_data(self, collection_name, partition_name) -> pd.DataFrame:
        collection_info = self._src_client.describe_collection(collection_name)
        collection_id = collection_info["collection_id"]
        field_infos = collection_info["fields"]
        partition_id = self._src_client.get_partition_id(
            collection_name, partition_name
        )

        bin_log_path = os.path.join(str(collection_id), str(partition_id))
        data = self._get_storage_data(bin_log_path, field_infos)
        return pd.DataFrame(data)

    def _write_milvus_data(self, path, schema: CollectionSchema, data: pd.DataFrame):
        bulk_writer = LocalBulkWriter(
            schema=schema,
            local_path=path,
            segment_size=100 * 1024 * 1024,  # 100MB default value
            file_type=BulkFileType.PARQUET,
        )

        for _, row in data.iterrows():
            d = {}
            for field in schema.fields:
                if field.auto_id:
                    continue
                d[field.name] = (
                    list(row[field.name])
                    if field.dtype == DataType.FLOAT_VECTOR
                    else row[field.name]
                )
            bulk_writer.append_row(d)
        bulk_writer.commit()

    def _import_to_milvus(self, collection, path: str):
        # check if directory
        logging.info(f"Importing data to milvus, path: {path}")
        files = glob.glob(f"{path}/*/*.parquet")
        # add prefix to files
        files = [os.path.join(self._dest_bucket.root_dir, f) for f in files]
        logging.info(f"Importing data to milvus, files: {files}")
        job_id = self._dest_client.import_data(collection, files)

        return [job_id]

    def _validate_milvus(self, df: pd.DataFrame, job_ids: list[str]):
        # groupby fsr_df by vector_store_id and sum by face_reference_id
        logging.info("Validating data in milvus...")
        job_sum = {}
        for job_id in job_ids:
            # wait until job is done
            state = "Pending"
            while state != "Completed":
                data = self._dest_client.get_import_job(job_id)["data"]
                if data["state"] == "Failed":
                    raise Exception(f"Import job {job_id} failed")
                if data["state"] == "Completed":
                    job_sum[data["collectionName"]] = (job_id, data["importedRows"])
                    break
                state = data["state"]

        expect = len(df)
        actual = job_sum.get(collection, 0)[1]

        logging.info(
            f"Validating data in milvus, collection: {collection}, expect: {expect}, actual: {actual}"
        )
        if expect != actual:
            logging.error(
                f"Data in milvus is not correct, jobId: {job_id}, collection: {collection}, expect: {expect}, actual: {actual}"
            )
            raise Exception(
                f"Data in milvus is not correct, jobId: {job_id}, collection: {collection}, expect: {expect}, actual: {actual}"
            )
        logging.info("Data in milvus is correct")

    def _migrate_partition(self, collection: str, partition: str):
        data_dir = os.path.join("data", collection.lower(), partition)
        os.makedirs(data_dir, exist_ok=True)

        # check if migration has been done
        if self._dest_bucket.object_exists(f"{data_dir}/success.txt"):
            logging.warning(f"Migration has been done, skip {collection}-{partition}")
            return

        if not partition:
            logging.error("Partition name is required")
            return

        if not self._src_client.has_collection(collection):
            logging.error(f"Collection {collection} not found")
            return
        if not self._src_client.has_partition(collection, partition):
            logging.error(f"Partition {partition} not found")
            return

        logging.info(
            f"Start migrating collection: {collection}, partition, {partition}"
        )
        logging.debug(f"Reading data from storage...")
        raw_file = f"{data_dir}/raw.pkl"
        if os.path.exists(raw_file):
            df = pd.read_pickle(raw_file)
        else:
            df = self._read_storage_data(collection, partition)
            if not df.empty:
                df.to_pickle(raw_file)

        if df.empty:
            logging.info(f"Empty data, skip {collection}-{partition}")
            return

        logging.info(
            f"Read {len(df)} records, collection: {collection}, partition: {partition} "
        )

        schema = self._src_client.parse_schema(collection)
        self._write_milvus_data(data_dir, schema, df)

        path = f"{data_dir}"
        try:
            files = glob.glob(f"{path}/*/*.parquet")
            # upload all files to oss
            for file in files:
                logging.info(f"Uploading file to oss: {file}")
                with open(file, "rb") as f:
                    self._dest_bucket.put_object(file, f.read())

            job_ids = self._import_to_milvus(collection, path)
            self._validate_milvus(df, job_ids)

            self._dest_bucket.put_object(
                f"{data_dir}/success.txt", bytes("", encoding="utf-8")
            )
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        finally:
            # delete all files when done
            shutil.rmtree(path)

    def migrate_collection_definition(self, collection: str):
        # Step 1: Check if the collection exists in the source Milvus
        if not self._src_client.has_collection(collection):
            raise ValueError(
                f"Collection {collection} does not exist in the source Milvus"
            )

        # Step 2: Create the collection in the destination Milvus if not exist
        if self._dest_client.has_collection(collection):
            raise ValueError(
                f"Collection {collection} already exist in the target Milvus"
            )

        logging.info(f"Creating collection, collection: {collection}")
        self._migration_schema(collection)
        self._migration_index(collection)

    def migrate(
        self,
        collection: str,
        partitions: list = None,
        ignore_partitions: list = None,
        mapping_schema: bool = False,
    ):
        if mapping_schema:
            self.migrate_collection_definition(collection)

        if ignore_partitions is None or len(ignore_partitions) == 0:
            ignore_partitions = []

        if partitions is None or len(partitions) == 0:
            partitions = self._src_client.list_partitions(collection)

        if partitions is None or len(partitions) == 0:
            logging.warning(f"No partitions found in collection: {collection}")
            return

        for partition in partitions:
            if partition in ignore_partitions:
                logging.info(f"Ignore partition: {partition}")
                continue
            self._migrate_partition(collection, partition)


def build_migration_tool() -> MilvusMigrationTool:
    import os
    from dotenv import load_dotenv

    load_dotenv()

    source_milvus = ConnectParams(
        alias="source",
        host=os.environ.get("MILVUS_HOST_SOURCE", "localhost"),
        port=os.environ.get("MILVUS_PORT_SOURCE", 19530),
        username=os.environ.get("MILVUS_USERNAME_SOURCE", ""),
        password=os.environ.get("MILVUS_PASSWORD_SOURCE", ""),
        database=os.environ.get("MILVUS_DATABASE_SOURCE", "default"),
    )

    dest_milvus = ConnectParams(
        alias="dest",
        host=os.environ.get("MILVUS_HOST_DEST", "localhost"),
        port=os.environ.get("MILVUS_PORT_DEST", 19530),
        username=os.environ.get("MILVUS_USERNAME_DEST", ""),
        password=os.environ.get("MILVUS_PASSWORD_DEST", ""),
        database=os.environ.get("MILVUS_DATABASE_DEST", "default"),
    )

    source_oss = OssConnectParam(
        endpoint=os.environ.get("OSS_ENDPOINT_SOURCE", None),
        access_key=os.environ.get("OSS_ACCESS_KEY_SOURCE", None),
        secret_key=os.environ.get("OSS_SECRET_KEY_SOURCE", None),
        bucket_name=os.environ.get("OSS_BUCKET_SOURCE", None),
        root_dir=os.environ.get("OSS_ROOT_DIR_SOURCE", ""),
    )

    dest_oss = OssConnectParam(
        endpoint=os.environ.get("OSS_ENDPOINT_DEST", None),
        access_key=os.environ.get("OSS_ACCESS_KEY_DEST", None),
        secret_key=os.environ.get("OSS_SECRET_KEY_DEST", None),
        bucket_name=os.environ.get("OSS_BUCKET_DEST", None),
        root_dir=os.environ.get("OSS_ROOT_DIR_DEST", ""),
    )

    return MilvusMigrationTool(source_milvus, dest_milvus, source_oss, dest_oss)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--collection", required=True, help="Collection name (required)"
    )
    parser.add_argument(
        "--schema",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="Mapping schema",
    )
    parser.add_argument("--partitions", required=False, help="Partition names")
    parser.add_argument("--ignore-partitions", required=False, help="Ignore partitions")

    args = parser.parse_args()

    collection = args.collection
    partitions = [] if not args.partitions else args.partitions.split(",")
    ignore_partitions = (
        [] if not args.ignore_partitions else args.ignore_partitions.split(",")
    )
    mapping_schema = args.schema

    migration_tool = build_migration_tool()
    migration_tool.migrate(collection, partitions, ignore_partitions, mapping_schema)
