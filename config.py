"""Configuration: Streamlit secrets first, then environment / .env. Never logs values."""
import os

from dotenv import load_dotenv

load_dotenv()


def get(name: str, default: str = "") -> str:
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


def amd_settings() -> dict:
    return {
        "api_key": get("AMD_API_KEY"),
        "base_url": get("AMD_BASE_URL").rstrip("/"),
        "model": get("AMD_MODEL"),
    }


def is_configured() -> bool:
    s = amd_settings()
    return bool(s["api_key"] and s["base_url"] and s["model"])
