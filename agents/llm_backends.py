# file: agents/llm_backends.py
from __future__ import annotations
"""
LLM backends for producing high-level trading Intents.

This module is OPTIONAL. If no API keys are present, you can ignore it and the
LLMDecisionAgent will fall back to a deterministic heuristic.

Backends
--------
- OpenAIChatBackend: uses OpenAI Chat Completions API.
  env:
    OPENAI_API_KEY
    OPENAI_MODEL (default: gpt-4o-mini)

- GeminiBackend: uses Google Generative AI (Gemini 1.5).
  env:
    GEMINI_API_KEY
    GEMINI_MODEL (default: gemini-1.5-flash)

- DeepSeekBackend: uses DeepSeek's OpenAI-compatible API.
  env:
    DEEPSEEK_API_KEY
    DEEPSEEK_MODEL (default: deepseek-chat)

- ChatGLMBackend: uses ZhipuAI's ChatGLM API.
  env:
    CHATGLM_API_KEY
    CHATGLM_MODEL (default: glm-4-flash)

- resolve_llm_backend(): returns the first available backend or None.

Contract
--------
Each backend implements:
    propose_intent(obs: Observation) -> Intent

It must be robust to occasional failures and return a *sane* Intent fallback.
"""
import os
import re
from typing import Optional, Tuple

import orjson

from agents.base import Intent, Observation


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _parse_json_loose(text: str) -> dict:
    """
    Extract the first JSON object from text and parse with orjson.
    If nothing found, raise ValueError.
    """
    # Try to find a {...} block (non-greedy)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found")
    raw = m.group(0)
    return orjson.loads(raw)


def _intent_from_dict(d: dict) -> Intent:
    tgt = float(d.get("target_position", 0.0))
    urg = _clamp(float(d.get("urgency", 0.5)), 0.0, 1.0)
    ms = float(d.get("max_spread", 0.0025))
    th = int(d.get("time_horizon", 20))
    note = d.get("note", None)
    return Intent(target_position=tgt, urgency=urg, max_spread=ms, time_horizon=max(1, th), note=note)


def _prompt_from_obs(obs: Observation) -> str:
    """
    Minimal instruction to elicit a structured JSON intent.
    Keep it short to control cost; the simulator already provides key state.
    """
    mid = obs.mid if obs.mid is not None else obs.last_trade
    spread = obs.spread if obs.spread is not None else 0.0
    inv = obs.account.inventory
    cash = obs.account.cash
    bid_depth = obs.bid_depth or 0.0
    ask_depth = obs.ask_depth or 0.0

    return f"""
You are a trading decision module for a simulated market. 
Given state, return a single JSON object with fields:
  target_position (float, absolute units), urgency (0..1), max_spread (relative, e.g., 0.002),
  time_horizon (int ticks), note (short string).

State:
  time: {obs.time}
  mid: {mid}
  spread: {spread}
  inventory: {inv}
  cash: {cash}
  bid_depth: {bid_depth}
  ask_depth: {ask_depth}

Guidelines:
- Keep risk modest; avoid extreme swings unless depth is high and spread is tight.
- Urgency high (>0.7) when spread is tight and a clear imbalance exists; otherwise moderate (~0.4-0.6).
- Target small inventory when uncertain (between -2 and 2).
- JSON ONLY. No markdown. No extra text.

Example:
{{"target_position": 1.0, "urgency": 0.6, "max_spread": 0.002, "time_horizon": 20, "note": "mild long tilt"}}
    """.strip()


# ---------------------------
# OpenAI Chat Completions API
# ---------------------------

class OpenAIChatBackend:
    def __init__(self, model: Optional[str] = None) -> None:
        from openai import OpenAI  # lazy import
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def propose_intent(self, obs: Observation) -> Intent:
        try:
            prompt = _prompt_from_obs(obs)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You output ONLY compact JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=120,
            )
            text = resp.choices[0].message.content or "{}"
            data = _parse_json_loose(text)
            return _intent_from_dict(data)
        except Exception:
            # Safe fallback
            return Intent(target_position=0.0, urgency=0.5, max_spread=0.0025, time_horizon=20, note="openai-fallback")


# -------------
# Gemini Backend
# -------------

class GeminiBackend:
    def __init__(self, model: Optional[str] = None) -> None:
        import google.generativeai as genai  # lazy import
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        self.genai = genai
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.model = genai.GenerativeModel(self.model_name)

    def propose_intent(self, obs: Observation) -> Intent:
        try:
            prompt = _prompt_from_obs(obs)
            resp = self.model.generate_content(prompt)
            text = resp.text or "{}"
            data = _parse_json_loose(text)
            return _intent_from_dict(data)
        except Exception:
            return Intent(target_position=0.0, urgency=0.5, max_spread=0.0025, time_horizon=20, note="gemini-fallback")


# --------------------
# DeepSeek (OpenAI API)
# --------------------

class DeepSeekBackend:
    def __init__(self, model: Optional[str] = None) -> None:
        from openai import OpenAI  # lazy import

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")

        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    def propose_intent(self, obs: Observation) -> Intent:
        try:
            prompt = _prompt_from_obs(obs)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You output ONLY compact JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=120,
            )
            text = resp.choices[0].message.content or "{}"
            data = _parse_json_loose(text)
            return _intent_from_dict(data)
        except Exception:
            return Intent(target_position=0.0, urgency=0.5, max_spread=0.0025, time_horizon=20, note="deepseek-fallback")


# --------------
# ChatGLM Backend
# --------------

class ChatGLMBackend:
    def __init__(self, model: Optional[str] = None) -> None:
        import httpx  # lazy import

        api_key = os.getenv("CHATGLM_API_KEY")
        if not api_key:
            raise RuntimeError("CHATGLM_API_KEY not set")

        base_url = os.getenv("CHATGLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        self.client = httpx.Client(base_url=base_url, timeout=15.0)
        self.api_key = api_key
        self.model = model or os.getenv("CHATGLM_MODEL", "glm-4-flash")

    def propose_intent(self, obs: Observation) -> Intent:
        try:
            prompt = _prompt_from_obs(obs)
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You output ONLY compact JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 120,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            resp = self.client.post("/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            parsed = _parse_json_loose(text)
            return _intent_from_dict(parsed)
        except Exception:
            return Intent(target_position=0.0, urgency=0.5, max_spread=0.0025, time_horizon=20, note="chatglm-fallback")


# -------------------------
# Backend resolution helper
# -------------------------

def resolve_llm_backend() -> Optional[object]:
    """
    Return an instantiated backend based on available environment variables.
    Priority: OpenAI -> Gemini -> DeepSeek -> ChatGLM. Returns None if none configured.
    """
    try:
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIChatBackend()
    except Exception:
        pass
    try:
        if os.getenv("GEMINI_API_KEY"):
            return GeminiBackend()
    except Exception:
        pass
    try:
        if os.getenv("DEEPSEEK_API_KEY"):
            return DeepSeekBackend()
    except Exception:
        pass
    try:
        if os.getenv("CHATGLM_API_KEY"):
            return ChatGLMBackend()
    except Exception:
        pass
    return None
