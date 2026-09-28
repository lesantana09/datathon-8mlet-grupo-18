import os
from io import BytesIO

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from core.config import settings
from core.logging import setup_logging

logger = setup_logging()


class StorageClient:
    """
    Cliente que abstrai as operações de leitura e escrita no Data Lake (MinIO/AWS S3).

    Args:
        region_name (str):
            Região da AWS.
        endpoint_url (str):
            URL do MinIO.
        access_key (str):
            Chave de acesso do MinIO.
        secret_key (str):
            Chave secreta do MinIO.
        s3_client (boto3.client):
            Cliente S3.
    """

    def __init__(self):
        """
        Inicializa o cliente S3.
        """
        self.region_name = settings.AWS_REGION
        self.endpoint_url = settings.AWS_ENDPOINT_URL
        self.access_key = settings.DAGSHUB_APP_TOKEN
        self.secret_key = settings.DAGSHUB_APP_TOKEN
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region_name,
        )

    def download(self, local_path: str, bucket: str, file: str):
        """
        Baixa um ficheiro do Data Lake.

        Args:
            local_path (str): Caminho local onde o ficheiro será guardado.
            bucket (str): Nome do bucket.
            file (str): Nome do ficheiro.

        Returns:
            bool: True se o download foi bem-sucedido, False caso contrário.

        Raises:
            ClientError: Se ocorrer um erro durante o download.
        """
        try:
            logger.debug(
                f"Iniciando download de s3://{bucket}/{file} para {local_path}..."
            )
            os.makedirs(local_path, exist_ok=True)
            if not self.file_exists(bucket, file):
                logger.error(f"Ficheiro s3://{bucket}/{file} NÃO existe")
                return False
            self.s3_client.download_file(bucket, file, f"{local_path}/{file}")
            logger.debug("Download concluído com sucesso!")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.error(f"Ficheiro {local_path}/{file} NÃO existe")
                return False
            # Qualquer outro erro (403 permissões, rede, etc.)
            raise e

    def download_bytes(self, bucket: str, file: str):
        """
        Baixa um ficheiro do Data Lake.

        Args:
            local_path (str): Caminho local onde o ficheiro será guardado.
            bucket (str): Nome do bucket.
            file (str): Nome do ficheiro.

        Returns:
            bool: True se o download foi bem-sucedido, False caso contrário.

        Raises:
            ClientError: Se ocorrer um erro durante o download.
        """
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=file)
            data = response["Body"].read()
            logger.debug(
                "Objeto baixado: s3://%s/%s (%d bytes).", bucket, file, len(data)
            )
            return data
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                logger.error("Objeto não encontrado: s3://%s/%s.", bucket, file)
                raise FileNotFoundError(
                    f"Objeto não encontrado: s3://{bucket}/{file}"
                ) from e
            logger.error("Falha ao baixar s3://%s/%s: %s", bucket, file, e)
            raise

    def upload(self, local_path: str, bucket: str, file: str):
        """
        Faz o upload de um ficheiro para o Data Lake.

        Args:
            local_path (str): Caminho local de onde o ficheiro será enviado.
            bucket (str): Nome do bucket.
            file (str): Nome do ficheiro.

        Returns:
            bool: True se o upload foi bem-sucedido, False caso contrário.

        Raises:
            ClientError: Se ocorrer um erro durante o upload.
        """
        try:
            logger.debug(
                f"Iniciando upload de {local_path}/{file} para s3://{bucket}/{file}..."
            )
            self.s3_client.upload_file(f"{local_path}/{file}", bucket, file)

            logger.debug("Validando a integridade do arquivo no bucket S3...")
            response = self.s3_client.head_object(Bucket=bucket, Key=file)
            file_size_mb = response["ContentLength"] / (1024 * 1024)
            logger.debug(
                f"Validação com sucesso! Tamanho do arquivo: {file_size_mb:.2f} MB"
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.error(f"Ficheiro {local_path}/{file} NÃO existe")
                return False
            # Qualquer outro erro (403 permissões, rede, etc.)
            raise e

    def upload_bytes(self, data: bytes, bucket: str, file: str):
        """
        Faz o upload de um ficheiro para o Data Lake.

        Args:
            data (bytes): Dados a serem enviados.
            bucket (str): Nome do bucket.
            file (str): Nome do ficheiro.

        Returns:
            bool: True se o upload foi bem-sucedido, False caso contrário.

        Raises:
            ClientError: Se ocorrer um erro durante o upload.
        """
        try:
            logger.debug(f"Iniciando upload de {data} para s3://{bucket}/{file}...")
            self.s3_client.put_object(Bucket=bucket, Key=file, Body=data)

            logger.debug("Validando a integridade do arquivo no bucket S3...")
            response = self.s3_client.head_object(Bucket=bucket, Key=file)
            file_size_mb = response["ContentLength"] / (1024 * 1024)
            logger.debug(
                f"Validação com sucesso! Tamanho do arquivo: {file_size_mb:.2f} MB"
            )
            return True
        except ClientError as e:
            # Qualquer outro erro (403 permissões, rede, etc.)
            raise e

    def save_parquet(self, df: pd.DataFrame, bucket: str, file_path: str) -> str:
        """
        Salva um DataFrame num bucket em formato Parquet.

        Args:
            df (pd.DataFrame): DataFrame a ser salvo.
            bucket (str): Nome do bucket.
            file_path (str): Caminho do ficheiro no bucket.
        """
        try:
            # Escreve o parquet na memória em vez de criar um ficheiro no disco local
            parquet_buffer = BytesIO()
            df.to_parquet(parquet_buffer, index=False)

            self.s3_client.put_object(
                Bucket=bucket, Key=file_path, Body=parquet_buffer.getvalue()
            )
            logger.debug(f"Ficheiro guardado com sucesso: s3://{bucket}/{file_path}")
            return f"s3://{bucket}/{file_path}"
        except Exception as e:
            logger.error(f"Erro ao guardar ficheiro no S3: {e}")
            raise e

    def load_parquet(self, bucket: str, file_path: str) -> pd.DataFrame:
        """
        Carrega um ficheiro Parquet do Data Lake para um DataFrame.

        Args:
            bucket (str): Nome do bucket.
            file_path (str): Caminho do ficheiro no bucket.

        Returns:
            pd.DataFrame: DataFrame carregado do bucket.
        """
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=file_path)
            parquet_content = response["Body"].read()
            return pd.read_parquet(BytesIO(parquet_content))
        except Exception as e:
            logger.error(f"Erro ao ler ficheiro do S3: {e}")
            raise e

    def file_exists(self, bucket: str, file_path: str) -> bool:
        """
        Verifica se um ficheiro já existe no bucket do Data Lake.

        Args:
            bucket (str): Nome do bucket.
            file_path (str): Caminho do ficheiro no bucket.

        Returns:
            bool: True se o ficheiro existe, False caso contrário.
        """
        try:
            self.s3_client.head_object(Bucket=bucket, Key=file_path)

            logger.debug(f"Ficheiro s3://{bucket}/{file_path} existe")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.debug(f"Ficheiro s3://{bucket}/{file_path} NÃO existe")
                return False
            logger.error(f"Erro ao verificar ficheiro no S3: {e}")
            raise e

    def folder_exists(self, bucket: str, folder_path: str) -> bool:
        """
        Verifica se uma pasta já existe no bucket do Data Lake.

        Args:
            bucket (str): Nome do bucket.
            folder_path (str): Caminho da pasta no bucket.

        Returns:
            bool: True se a pasta existe, False caso contrário.
        """
        try:
            folder_path = folder_path.rstrip("/")
            self.s3_client.list_objects(
                Bucket=bucket, Prefix=folder_path, Delimiter="/", MaxKeys=1
            )
            logger.debug(f"Pasta s3://{bucket}/{folder_path} existe")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.debug(f"Pasta s3://{bucket}/{folder_path} NÃO existe")
                return False
            logger.error(f"Erro ao verificar pasta no S3: {e}")
            raise e

    def delete_parquet(self, bucket: str, file_path: str) -> str:
        """
        Apaga um ficheiro Parquet do Data Lake.

        Args:
            bucket (str): Nome do bucket.
            file_path (str): Caminho do ficheiro no bucket.
        """
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=file_path)
            logger.debug(f"Ficheiro apagado com sucesso: s3://{bucket}/{file_path}")
            return f"s3://{bucket}/{file_path}"
        except Exception as e:
            logger.error(f"Erro ao apagar ficheiro no S3: {e}")
            raise e


if __name__ == "__main__":
    storage = StorageClient()
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    print(df.to_csv())
    print("Save parquet: ", storage.save_parquet(df, "datathon-8mlet-grupo-18", "test.parquet"))
    print("Arquivo existe? ", storage.file_exists("datathon-8mlet-grupo-18", "test.parquet"))
    print("Delete parquet: ", storage.delete_parquet("datathon-8mlet-grupo-18", "test.parquet"))
    print("Arquivo existe? ", storage.file_exists("datathon-8mlet-grupo-18", "test.parquet"))
