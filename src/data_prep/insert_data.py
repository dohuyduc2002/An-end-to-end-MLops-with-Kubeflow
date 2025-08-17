import argparse
import os
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

    temp_file_path = "/tmp/tmp.csv"

    stat = minio_client.stat_object(args.bucket, args.file)
    print(
        f"Downloading {args.file} ({stat.size / (1024*1024):.2f} MB) from MinIO in one go...",
        flush=True,
    )

    resp = minio_client.get_object(args.bucket, args.file)
    data = resp.read() 
    resp.close()
    resp.release_conn()

    with open(temp_file_path, "wb") as f:
        f.write(data)
    print(f"✔ Download completed: {temp_file_path}", flush=True)

    table_fq = f"{args.schema}.{args.table}"
    conn = postgres_client.create_conn()
    conn.autocommit = False
    cur = conn.cursor()

    print(f"Loading {temp_file_path} -> {table_fq}", flush=True)
    with open(temp_file_path, "r", encoding="utf-8") as f:
        cur.copy_expert(
            f"COPY {table_fq} FROM STDIN WITH (FORMAT CSV, HEADER TRUE)",
            f,
        )

    conn.commit()
    cur.close()
    conn.close()
    print("✔ Data loaded into PostgreSQL", flush=True)

    # Xóa file tạm
    try:
        os.remove(temp_file_path)
        print("🗑 Temp file removed", flush=True)
    except OSError as e:
        print(f"⚠ Could not delete temp file: {e}", flush=True)


if __name__ == "__main__":
    main()
