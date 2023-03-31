pip install mlflow==1.25.1 psycopg2 protobuf==3.20.* boto3

docker run -d --rm \
	--name psql-for-ml \
	-e POSTGRES_PASSWORD=mlflow \
	-e POSTGRES_USER=mlflow \
	-e POSTGRES_DB=mlflow \
	-e PGDATA=/var/lib/postgresql/data/pgdata \
	-v ./psql-data:/var/lib/postgresql/data \
	-p 5432:5432 \
	postgres:15

docker run -d --rm \
  --name minio-container \
  -p 9000:9000 \
  -p 9001:9001 \
  -v ./minio-data:/data \
  minio/minio server /data --console-address ":9001"

export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
export MLFLOW_S3_IGNORE_TLS=true
export BACKEND_URI=postgresql://mlflow:mlflow@127.0.0.1:5432/mlflow
export ARTIFACT_ROOT=s3://mlflow/mlartifacts  
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

mlflow server \
  --backend-store-uri ${BACKEND_URI} \
  --artifacts-destination ${ARTIFACT_ROOT} \
  --serve-artifacts \
  --host 0.0.0.0
  
