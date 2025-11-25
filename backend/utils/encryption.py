"""
Encryption utilities for sensitive data (API keys)
"""

import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


def _get_fernet() -> Fernet:
    """
    Get Fernet cipher instance using app encryption key
    """
    # Derive a proper Fernet key from the encryption key
    key = settings.ENCRYPTION_KEY.encode()
    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"vokg_salt_v1",  # Static salt for consistent key derivation
        iterations=100000,
    )
    derived_key = base64.urlsafe_b64encode(kdf.derive(key))
    return Fernet(derived_key)


def encrypt_api_key(api_key: str) -> str:
    """
    Encrypt an API key for storage

    Args:
        api_key: Plain text API key

    Returns:
        Encrypted API key as base64 string
    """
    try:
        fernet = _get_fernet()
        encrypted = fernet.encrypt(api_key.encode())
        return encrypted.decode()
    except Exception as e:
        logger.error(f"Failed to encrypt API key: {e}")
        raise


def decrypt_api_key(encrypted_key: str) -> str:
    """
    Decrypt an encrypted API key

    Args:
        encrypted_key: Encrypted API key

    Returns:
        Plain text API key
    """
    try:
        fernet = _get_fernet()
        decrypted = fernet.decrypt(encrypted_key.encode())
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Failed to decrypt API key: {e}")
        raise
