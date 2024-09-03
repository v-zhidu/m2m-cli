import logging
import os
import oss2


class OssConnectParam:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        root_dir: str = "",
    ):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.root_dir = root_dir


class OssStorage:
    def __init__(self, param: OssConnectParam):
        self.root_dir = param.root_dir
        self._bucket = self._init_oss_bucket(
            param.endpoint, param.access_key, param.secret_key, param.bucket_name
        )

    def _init_oss_bucket(self, endpoint, access_key, secret_key, bucket):
        auth = oss2.Auth(access_key, secret_key)
        return oss2.Bucket(auth, endpoint, bucket)

    def list_objects(self, prefix: str = None, recursive=False):
        prefix = prefix.strip("/")
        prefix += "/"
        if self.root_dir:
            prefix = os.path.join(self.root_dir, prefix)
        logging.info(f"Listing objects in OSS bucket with prefix: {prefix}")
        return [
            obj.key.replace(self.root_dir, "")
            for obj in oss2.ObjectIterator(
                self._bucket, prefix=prefix, delimiter="/" if not recursive else ""
            )
        ]

    def download_object(self, key: str):
        logging.debug(f"Getting object from OSS bucket with key: {key}")
        key = os.path.join(self.root_dir, key)
        return self._bucket.get_object(key).read()

    def put_object(self, key: str, data: bytes):
        logging.debug(f"Putting object to OSS bucket with key: {key}")
        key = os.path.join(self.root_dir, key)
        return self._bucket.put_object(key, data)

    def object_exists(self, key: str):
        if not key:
            return False
        logging.debug(f"Checking if object exists in OSS bucket with key: {key}")
        key = os.path.join(self.root_dir, key)
        return self._bucket.object_exists(key)


def create_oss_storage(params: list[OssConnectParam]) -> list[OssStorage]:
    return [OssStorage(param) for param in params]
