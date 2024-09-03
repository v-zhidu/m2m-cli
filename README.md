# m2m-cli

## Usage

### Set Environment Variables

```bash
MILVUS_HOST_SOURCE=localhost
MILVUS_PORT_SOURCE=19530
MILVUS_USERNAME_SOURCE=root
MILVUS_PASSWORD_SOURCE=
MILVUS_DATABASE_SOURCE=default

MILVUS_HOST_DEST=localhost
MILVUS_PORT_DEST=19530
MILVUS_USERNAME_DEST=root
MILVUS_PASSWORD_DEST=
MILVUS_DATABASE_DEST=default

OSS_ENDPOINT_SOURCE=
OSS_ACCESS_KEY_SOURCE=
OSS_SECRET_KEY_SOURCE=
OSS_BUCKET_SOURCE=
OSS_ROOT_DIR_SOURCE=

OSS_ENDPOINT_DEST=
OSS_ACCESS_KEY_DEST=
OSS_SECRET_KEY_DEST=
OSS_BUCKET_DEST=
OSS_ROOT_DIR_DEST=
```

### CLI

```txt
usage: main.py [-h] --collection COLLECTION [--schema | --no-schema] [--partitions PARTITIONS] [--ignore-partitions IGNORE_PARTITIONS]

options:
  -h, --help            show this help message and exit
  --collection COLLECTION
                        Collection name (required)
  --schema, --no-schema
                        Mapping schema
  --partitions PARTITIONS
                        Partition names
  --ignore-partitions IGNORE_PARTITIONS
                        Ignore partitions
```

## Build Docker Image

```bash
pack build m2m-cli --buildpack paketo-buildpacks/python --builder paketobuildpacks/builder-jammy-base -t m2m-cli:latest
```