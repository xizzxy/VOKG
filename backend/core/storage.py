"""
Storage abstraction layer for S3/MinIO
Handles video, frame, mask, and embedding storage
"""

import io
import json
from pathlib import Path
from typing import Optional, BinaryIO, Union
from datetime import timedelta

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from .config import settings
from .logging import get_logger

logger = get_logger(__name__)


class StorageClient:
    """
    Unified storage client for S3/MinIO operations
    """

    def __init__(self):
        """
        Initialize S3/MinIO client
        """
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT_URL if settings.STORAGE_TYPE == "minio" else None,
            aws_access_key_id=settings.S3_ACCESS_KEY_ID,
            aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
            region_name=settings.S3_REGION,
            config=Config(signature_version="s3v4"),
        )
        self.bucket_name = settings.S3_BUCKET_NAME
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """
        Create bucket if it doesn't exist
        """
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Storage bucket '{self.bucket_name}' exists")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                logger.info(f"Creating storage bucket: {self.bucket_name}")
                try:
                    if settings.S3_REGION == "us-east-1":
                        self.client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={"LocationConstraint": settings.S3_REGION},
                        )
                    logger.info(f"Bucket '{self.bucket_name}' created successfully")
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")
                    raise
            else:
                logger.error(f"Error checking bucket: {e}")
                raise

    def upload_file(
        self,
        file_path: Union[str, Path],
        object_key: str,
        metadata: Optional[dict] = None,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload a file to storage

        Args:
            file_path: Local file path
            object_key: S3 object key (path in bucket)
            metadata: Optional metadata dict
            content_type: MIME type

        Returns:
            Object key
        """
        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = {k: str(v) for k, v in metadata.items()}
            if content_type:
                extra_args["ContentType"] = content_type

            self.client.upload_file(
                str(file_path), self.bucket_name, object_key, ExtraArgs=extra_args or None
            )
            logger.debug(f"Uploaded file to {object_key}", file_path=str(file_path))
            return object_key
        except ClientError as e:
            logger.error(f"Failed to upload file: {e}", object_key=object_key)
            raise

    def upload_fileobj(
        self,
        file_obj: BinaryIO,
        object_key: str,
        metadata: Optional[dict] = None,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload file object to storage

        Args:
            file_obj: File-like object
            object_key: S3 object key
            metadata: Optional metadata
            content_type: MIME type

        Returns:
            Object key
        """
        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = {k: str(v) for k, v in metadata.items()}
            if content_type:
                extra_args["ContentType"] = content_type

            self.client.upload_fileobj(
                file_obj, self.bucket_name, object_key, ExtraArgs=extra_args or None
            )
            logger.debug(f"Uploaded file object to {object_key}")
            return object_key
        except ClientError as e:
            logger.error(f"Failed to upload file object: {e}", object_key=object_key)
            raise

    def download_file(self, object_key: str, file_path: Union[str, Path]) -> Path:
        """
        Download file from storage

        Args:
            object_key: S3 object key
            file_path: Local destination path

        Returns:
            Path to downloaded file
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(self.bucket_name, object_key, str(file_path))
            logger.debug(f"Downloaded {object_key} to {file_path}")
            return file_path
        except ClientError as e:
            logger.error(f"Failed to download file: {e}", object_key=object_key)
            raise

    def download_fileobj(self, object_key: str) -> io.BytesIO:
        """
        Download file to memory

        Args:
            object_key: S3 object key

        Returns:
            BytesIO object
        """
        try:
            file_obj = io.BytesIO()
            self.client.download_fileobj(self.bucket_name, object_key, file_obj)
            file_obj.seek(0)
            logger.debug(f"Downloaded {object_key} to memory")
            return file_obj
        except ClientError as e:
            logger.error(f"Failed to download file object: {e}", object_key=object_key)
            raise

    def delete_file(self, object_key: str) -> bool:
        """
        Delete file from storage

        Args:
            object_key: S3 object key

        Returns:
            True if successful
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=object_key)
            logger.debug(f"Deleted {object_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete file: {e}", object_key=object_key)
            raise

    def delete_prefix(self, prefix: str) -> int:
        """
        Delete all objects with given prefix

        Args:
            prefix: Object key prefix

        Returns:
            Number of objects deleted
        """
        try:
            objects = self.list_objects(prefix)
            if not objects:
                return 0

            delete_keys = [{"Key": obj} for obj in objects]
            response = self.client.delete_objects(
                Bucket=self.bucket_name, Delete={"Objects": delete_keys}
            )
            deleted_count = len(response.get("Deleted", []))
            logger.info(f"Deleted {deleted_count} objects with prefix {prefix}")
            return deleted_count
        except ClientError as e:
            logger.error(f"Failed to delete prefix: {e}", prefix=prefix)
            raise

    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> list[str]:
        """
        List objects with given prefix

        Args:
            prefix: Object key prefix
            max_keys: Maximum number of keys to return

        Returns:
            List of object keys
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=prefix, MaxKeys=max_keys
            )
            objects = [obj["Key"] for obj in response.get("Contents", [])]
            logger.debug(f"Listed {len(objects)} objects with prefix {prefix}")
            return objects
        except ClientError as e:
            logger.error(f"Failed to list objects: {e}", prefix=prefix)
            raise

    def get_presigned_url(
        self, object_key: str, expiration: int = 3600, method: str = "get_object"
    ) -> str:
        """
        Generate presigned URL for object access

        Args:
            object_key: S3 object key
            expiration: URL expiration in seconds
            method: S3 method (get_object or put_object)

        Returns:
            Presigned URL
        """
        try:
            url = self.client.generate_presigned_url(
                method,
                Params={"Bucket": self.bucket_name, "Key": object_key},
                ExpiresIn=expiration,
            )
            logger.debug(f"Generated presigned URL for {object_key}", method=method)
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}", object_key=object_key)
            raise

    def object_exists(self, object_key: str) -> bool:
        """
        Check if object exists

        Args:
            object_key: S3 object key

        Returns:
            True if exists
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=object_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def get_object_metadata(self, object_key: str) -> dict:
        """
        Get object metadata

        Args:
            object_key: S3 object key

        Returns:
            Metadata dict
        """
        try:
            response = self.client.head_object(Bucket=self.bucket_name, Key=object_key)
            return {
                "size": response["ContentLength"],
                "last_modified": response["LastModified"],
                "content_type": response.get("ContentType"),
                "metadata": response.get("Metadata", {}),
            }
        except ClientError as e:
            logger.error(f"Failed to get metadata: {e}", object_key=object_key)
            raise

    # Helper methods for VOKG-specific storage patterns

    def upload_video(self, video_id: str, file_path: Path) -> str:
        """Upload video file"""
        object_key = f"videos/{video_id}/{file_path.name}"
        return self.upload_file(file_path, object_key, content_type="video/mp4")

    def upload_frame(self, video_id: str, frame_number: int, file_path: Path) -> str:
        """Upload extracted frame"""
        object_key = f"frames/{video_id}/{frame_number:06d}.jpg"
        return self.upload_file(file_path, object_key, content_type="image/jpeg")

    def upload_mask(self, video_id: str, frame_number: int, object_id: int, mask_data: bytes) -> str:
        """Upload SAM mask"""
        object_key = f"masks/{video_id}/{frame_number:06d}/{object_id}.png"
        return self.upload_fileobj(io.BytesIO(mask_data), object_key, content_type="image/png")

    def upload_embeddings(self, video_id: str, embeddings_data: dict) -> str:
        """Upload object embeddings as JSON"""
        object_key = f"embeddings/{video_id}/embeddings.json"
        json_bytes = json.dumps(embeddings_data).encode("utf-8")
        return self.upload_fileobj(
            io.BytesIO(json_bytes), object_key, content_type="application/json"
        )

    def upload_graph_data(self, video_id: str, graph_data: dict) -> str:
        """Upload knowledge graph JSON"""
        object_key = f"graphs/{video_id}/graph.json"
        json_bytes = json.dumps(graph_data).encode("utf-8")
        return self.upload_fileobj(
            io.BytesIO(json_bytes), object_key, content_type="application/json"
        )


# Singleton instance
_storage_client: Optional[StorageClient] = None


def get_storage_client() -> StorageClient:
    """
    Get singleton storage client instance
    """
    global _storage_client
    if _storage_client is None:
        _storage_client = StorageClient()
    return _storage_client
