import argparse
import io
from minio import Minio
from postgres_client import PostgresSQLClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--dbname", default="homecredit")
    parser.add_argument("--host", default="postgres.database.svc.cluster.local")
    parser.add_argument("--port", default="5432")
    # minio
    parser.add_argument(
        "--minio-endpoint", default="minio.minio.svc.cluster.local:9000"
    )
    parser.add_argument("--minio-access-key", default="minio")
    parser.add_argument("--minio-secret-key", default="minio123")
    parser.add_argument("--bucket", default="sample-data")
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    # clients
    postgres_client = PostgresSQLClient(
        database=args.dbname,
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
    )
    minio_client = Minio(
        endpoint=args.minio_endpoint,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        secure=False,
    )

    conn = postgres_client.create_conn()
    conn.autocommit = False

    table_fq = f"{args.schema}.{args.table}"
    print(f"Loading {args.file} -> {table_fq}", flush=True)

    resp = minio_client.get_object(args.bucket, args.file)
    text_stream = io.TextIOWrapper(resp, encoding="utf-8")

    cur = conn.cursor()
    cur.copy_expert(
        f"COPY {table_fq} FROM STDIN WITH (FORMAT CSV, HEADER TRUE)",
        text_stream,
    )
    conn.commit()

    cur.close()
    text_stream.detach()
    resp.close()
    resp.release_conn()
    conn.close()
    print("✔ Done")


if __name__ == "__main__":
    main()
