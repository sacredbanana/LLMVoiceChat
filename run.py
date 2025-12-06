#!/usr/bin/env python3
"""
run.py

Simple voice-chat client for an LLM server.

Features:
 - Voice activity detection (RMS threshold)
 - Local transcription via whisper
 - Sends text to LLM Server (OpenAI-chat-compatible endpoint by default)
 - Fallback to LLM Server /api/generate if needed
 - Text-to-speech using Coqui/ElevenLabs/macOS TTS
 - Response editing (via --edit-responses and --edit-timeout)
"""
import argparse
import queue
import time
import sys
import io
import os
import re
import subprocess
import select
import sounddevice as sd
import numpy as np
import requests
import whisper
from dotenv import load_dotenv, find_dotenv
from TTS.api import TTS
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from typing import Optional
from contextlib import redirect_stdout


# Load environment variables from .env file
env_path = find_dotenv()
load_dotenv(env_path)

# ---------------------------
# Configuration (defaults)
# ---------------------------
REQUEST_TIMEOUT = os.getenv("REQUEST_TIMEOUT", 160.0)  # seconds

# ------------------------------------------------------------
# LLM Configuration
# ------------------------------------------------------------
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:1234")
LLM_MODEL = os.getenv("LLM_MODEL", None)
LLM_API_KEY = os.getenv("LLM_API_KEY", None)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", None)
SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE", None)
MAX_HISTORY = os.getenv("MAX_HISTORY", 1024)
EDIT_TIMEOUT = os.getenv("EDIT_TIMEOUT", 5.0)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Voice Input Configuration
# ------------------------------------------------------------
WHISPER_SAMPLE_RATE = os.getenv("WHISPER_SAMPLE_RATE", 16000)
WHISPER_CHANNELS = os.getenv("WHISPER_CHANNELS", 1)
WHISPER_CHUNK_DURATION = os.getenv("WHISPER_CHUNK_DURATION", 0.5)  # seconds per audio chunk (for VAD)
WHISPER_START_THRESHOLD = os.getenv("WHISPER_START_THRESHOLD", 0.01)  # RMS threshold to consider "speech started" (tune this)
WHISPER_SILENCE_TIMEOUT = os.getenv("WHISPER_SILENCE_TIMEOUT", 3.0)   # seconds of sustained silence to end utterance
WHISPER_MAX_UTTERANCE_DURATION = os.getenv("WHISPER_MAX_UTTERANCE_DURATION", 300.0)  # guard upper limit for a single utterance in seconds
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", None)
# ------------------------------------------------------------

# ------------------------------------------------------------
# TTS Configuration
# ------------------------------------------------------------
# General TTS Configuration
SPEECH_ENGINE = os.getenv("SPEECH_ENGINE", "coqui")
SPEECH_RATE = os.getenv("SPEECH_RATE", 1.0)

# Coqui
COQUI_MODEL = os.getenv("COQUI_TTS_MODEL", "tts_models/en/vctk/vits")
COQUI_SPEAKER = os.getenv("COQUI_TTS_SPEAKER", "p226")
COQUI_MODEL_PATH = os.getenv("COQUI_TTS_MODEL_PATH", None)
COQUI_CONFIG_PATH = os.getenv("COQUI_TTS_CONFIG_PATH", None)
COQUI_LANGUAGE = os.getenv("COQUI_TTS_LANGUAGE", None)
COQUI_DEVICE = os.getenv("COQUI_DEVICE", "cpu")
COQUI_SAMPLE_RATE = os.getenv("COQUI_SAMPLE_RATE", 22050)

# ElevenLabs
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", None)
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
ELEVENLABS_VOICE = os.getenv("ELEVENLABS_VOICE", "N2lVS1w4EtoT3dr4eOWO")

# macOS
MACOS_VOICE = os.getenv("MACOS_VOICE", None)
# ------------------------------------------------------------

# ---------------------------
# Utility: VAD-based recorder
# ---------------------------
class VoiceRecorder:
    """
    Records incoming microphone audio using a simple RMS-based VAD:
      - waits until RMS > start_threshold to start recording
      - stops when RMS drops below threshold for silence_timeout seconds
    Returns a NumPy array (float32) at sample_rate and mono.
    """
    def __init__(self,
                 sample_rate: int,
                 channels: int,
                 chunk_duration: float,
                 start_threshold: float,
                 silence_timeout: float,
                 max_utterance: float):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_frames = int(chunk_duration * sample_rate)
        self.dtype = 'float32'
        self.q = queue.Queue()
        self.start_threshold = start_threshold
        self.silence_timeout = silence_timeout
        self.max_utterance = max_utterance

        self.stream = None
        self._recording = False

    def _rms(self, block: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(block), axis=None)))

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # non-fatal info/warning
            print(f"[audio status] {status}", file=sys.stderr)
        # ensure mono shape
        if self.channels > 1:
            # convert to mono by averaging channels
            in_mono = np.mean(indata, axis=1)
        else:
            in_mono = indata[:, 0]
        # put a copy into queue
        self.q.put(in_mono.copy())

    def listen_once(self) -> Optional[np.ndarray]:
        """Blocks until a single utterance is captured and returns it as numpy float32 mono."""
        frames = []
        started = False
        silence_start_time = None
        utterance_start_time = time.time()

        with sd.InputStream(samplerate=self.sample_rate,
                            channels=self.channels,
                            dtype=self.dtype,
                            blocksize=self.chunk_frames,
                            callback=self._audio_callback):
            print("\n\nListening... (speak normally) Press Ctrl+C to quit.")
            while True:
                try:
                    chunk = self.q.get(timeout=1.0)
                except queue.Empty:
                    # no audio available — continue loop (possible if microphone warmup)
                    continue
                rms = self._rms(chunk)
                # debugging: uncomment for RMS display
                # print(f"rms={rms:.4f}")
                if not started:
                    if rms >= self.start_threshold:
                        started = True
                        frames.append(chunk)
                        silence_start_time = None
                        #print("Detected speech start")
                    else:
                        # ignore background noise
                        continue
                else:
                    frames.append(chunk)
                    if rms < self.start_threshold:
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        elif (time.time() - silence_start_time) >= self.silence_timeout:
                            # end of utterance
                            break
                    else:
                        silence_start_time = None

                # Guard: maximum utterance length
                if time.time() - utterance_start_time > self.max_utterance:
                    print("Reached maximum utterance duration, stopping recording.")
                    break

        if not frames:
            return None
        audio = np.concatenate(frames, axis=0)
        # ensure float32 in [-1.0, 1.0]
        audio = np.clip(audio, -1.0, 1.0).astype('float32')
        return audio

# ---------------------------
# Whisper transcription wrapper
# ---------------------------
class WhisperTranscriber:
    def __init__(self, model_name: str, device: str, language: str, sample_rate: int):
        print(f"Loading Whisper model '{model_name}' on {device} at {sample_rate} Hz with language '{language}' (this can take a while)...")
        self.model = whisper.load_model(model_name, device=device)
        self.sample_rate = sample_rate
        self.language = language
        print("Whisper loaded.")

    def transcribe_numpy(self, audio: np.ndarray) -> str:
        """
        Accepts a mono float32 numpy array at sample_rate.
        Writes to a temporary in-memory WAV and feeds whisper.
        """
        # Whisper's transcribe function can accept a numpy array plus sample rate.
        # But to be robust, use whisper's built-in preprocessing helper.
        try:
            kwargs = {}
            kwargs["fp16"] = False
            kwargs["task"] = "transcribe"
            if self.language:
                kwargs["language"] = self.language
            result = self.model.transcribe(audio, verbose=False, **kwargs)
            return result.get("text", "").strip()
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None

# ---------------------------
# LLM client (OpenAI-compatible & fallback)
# ---------------------------
class LLMClient:
    def __init__(self, base_url: str, model: str, timeout: float, system_prompt: Optional[str], max_history: int, api_key: Optional[str]):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.conversation_history = []  # List of {"role": "user/assistant", "content": "text"} dicts
        self.api_key = api_key
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _try_openai_chat(self, user_text: str) -> Optional[str]:
        """Try OpenAI-chat-compatible /v1/chat/completions endpoint."""
        url = f"{self.base_url}/v1/chat/completions"
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_text + "\n/no_think\n"})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "presence_penalty": 1.1,
            "top_p": 0.95,
            "top_logprobs": 40
        }
        try:
            r = self.session.post(url, json=payload, timeout=self.timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            # supports both 'choices' style or single 'output' style
            if "choices" in data and len(data["choices"]) > 0:
                # Chat completions response
                content = data["choices"][0].get("message", {}).get("content")
                if not content:
                    # some implementations put text under choices[0].text
                    content = data["choices"][0].get("text")
                return content
            # other response shapes: try 'output_text' or 'text'
            if "output" in data and isinstance(data["output"], list):
                return " ".join(map(str, data["output"]))
            for key in ("text", "output_text"):
                if key in data:
                    return data[key]
            return None
        except Exception as e:
            # print error and return None to try fallback
            print(f"[LLM] openai-chat attempt failed: {e}")
            return None

    def _try_llm_api(self, user_text: str) -> Optional[str]:
        """Try the server's /api/generate endpoint (common fallback)."""
        url = f"{self.base_url}/api/generate"
        
        # Build conversation prompt with history
        prompt_parts = []
        if self.system_prompt:
            prompt_parts.append(self.system_prompt)
            prompt_parts.append("")  # blank line
        
        # Add conversation history in a simple format
        for msg in self.conversation_history:
            if msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
        
        # Add current user message
        prompt_parts.append(f"User: {user_text}")
        prompt_parts.append("Assistant:")
        
        full_prompt = "\n".join(prompt_parts)
        
        payload = {
            "prompt": full_prompt,
            "max_new_tokens": 5120,
            # include other params if you like: temperature, top_p, etc.
        }
        try:
            r = self.session.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            # LLM Server's generate returns 'results' with 'text' often
            if isinstance(data, dict):
                # try 'text' or 'output' fields
                for key in ("text", "output", "results"):
                    if key in data:
                        v = data[key]
                        if isinstance(v, list):
                            # gather text fields
                            texts = []
                            for item in v:
                                if isinstance(item, dict):
                                    if "text" in item:
                                        texts.append(item["text"])
                                    elif "output" in item:
                                        texts.append(item["output"])
                                else:
                                    texts.append(str(item))
                            return "\n".join(t for t in texts if t)
                        elif isinstance(v, str):
                            return v
                # sometimes nested
                if "results" in data and isinstance(data["results"], list) and len(data["results"]) > 0:
                    first = data["results"][0]
                    if isinstance(first, dict) and "text" in first:
                        return first["text"]
            # fallback: return raw text
            return r.text
        except Exception as e:
            print(f"[LLM] /api/generate attempt failed: {e}")
            return None

    def chat(self, user_text: str) -> str:
        """Send user_text to the server; try OpenAI-chat endpoint first, then fallback."""
        # try openai chat-compatible endpoint
        resp = self._try_openai_chat(user_text)
        
        if not resp:
            raise RuntimeError("Failed to get response from the server (both endpoints failed). Check LLM_URL and that the server is running.")
        
        resp = resp.strip()
        
        # Add this exchange to conversation history
        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": resp})
        
        # Trim history if it exceeds max_history (keep pairs of user/assistant)
        if len(self.conversation_history) > self.max_history * 2:
            # Remove oldest pair (user + assistant messages)
            self.conversation_history = self.conversation_history[2:]
        
        return resp
    
    def chat_without_history(self, user_text: str) -> str:
        """
        Generate response without adding to conversation history.
        Used for regeneration attempts.
        """
        # try openai chat-compatible endpoint
        resp = self._try_openai_chat(user_text)
        
        if not resp:
            raise RuntimeError("Failed to get response from the server (both endpoints failed). Check LLM_URL and that the server is running.")
        
        return resp.strip()
    
    def add_to_history(self, user_text: str, assistant_response: str):
        """Manually add an exchange to conversation history."""
        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # Trim history if it exceeds max_history (keep pairs of user/assistant)
        if len(self.conversation_history) > self.max_history * 2:
            # Remove oldest pair (user + assistant messages)
            self.conversation_history = self.conversation_history[2:]

# ---------------------------
# Helpers for TTS text sanitization
# ---------------------------
def sanitize_text_for_tts(text: str, speech_rate: float) -> str:
    """
    Clean up model output for Coqui TTS to avoid crashes due to very short
    fragments, markdown bullets, emojis, or unsupported symbols.
    speech_rate: if < 1.0 (slower), add extra pauses between sentences
    """
    if not text:
        return ""
    # Normalize newlines
    t = re.sub(r"\r\n?|\r", "\n", text)
    # Remove Markdown bullet markers at the start of lines (*, -, +, •)
    t = re.sub(r"^\s*[\*\-\+\•]\s*", "", t, flags=re.MULTILINE)
    # Remove Markdown emphasis/code markers
    t = t.replace("**", "").replace("*", "").replace("`", "")
    # Strip URLs-in-markdown like [text](link)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)
    # Remove non-ASCII (including emojis) to keep model vocab happy
    t = re.sub(r"[^\x00-\x7F]+", " ", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    
    # Add pauses for slower speech by inserting commas/periods strategically
    if speech_rate < 1.0:
        # Add pauses between sentences for slower speech
        t = re.sub(r"([.!?])\s+", r"\1... ", t)
        # Add brief pauses after longer phrases (every 8-12 words)
        words = t.split()
        if len(words) > 8:
            new_words = []
            for i, word in enumerate(words):
                new_words.append(word)
                # Add pause every 8-12 words, but not at sentence boundaries
                if (i + 1) % 10 == 0 and not word.endswith(('.', '!', '?', '...')):
                    new_words.append(',')
            t = ' '.join(new_words)
    
    # Ensure there's some minimum length to prevent short-fragment issues
    if len(t) < 8:
        t = f"{t} Ok."
    # Ensure ending punctuation for prosody
    if t and t[-1] not in ".!?":
        t += "."
    return t

# ---------------------------
# Simple TTS (Coqui TTS - fully local)
# ---------------------------
class SimpleTTS:
    def __init__(self, model_name: str, model_path: str, config_path: str, speaker: str, language: Optional[str], device: str, speech_rate: float, sample_rate: int) -> None:
        """
        Uses Coqui TTS locally. You can change the model via --coqui-model or COQUI_MODEL.
        For multi-speaker or multilingual models, you can pass --coqui-speaker / --coqui-language.
        speech_rate: float, controls speech speed. 1.0 = normal, 0.5 = half speed (slower), 2.0 = double speed (faster) 
        """
        
        # Instantiate TTS; disable progress bar for cleaner console.
        use_gpu = (device.lower() == "cuda")
        stdout_capture = io.StringIO()
        # with redirect_stdout(stdout_capture):
        self.tts = TTS(model_name=model_name, model_path=model_path, config_path=config_path, progress_bar=False, gpu=use_gpu)
        self.speaker = speaker
        self.language = language
        self.speech_rate = speech_rate

        # Determine output sample rate from synthesizer
        # (available after model is loaded)
        self.sample_rate = getattr(self.tts.synthesizer, "output_sample_rate", sample_rate)

    def speak(self, text: str):
        if not text:
            return
        # Sanitize and avoid sentence auto-splitting into tiny fragments
        text_clean = sanitize_text_for_tts(text, self.speech_rate)
        # Build kwargs based on model capabilities
        kwargs = {}
        if self.speaker:
            kwargs["speaker"] = self.speaker
        if self.language:
            kwargs["language"] = self.language

        # Synthesize to waveform (numpy array)
        stdout_capture = io.StringIO()
        # with redirect_stdout(stdout_capture):
        try:
            try:
                wav = self.tts.tts(text=text_clean, split_sentences=False, **kwargs)
            except TypeError:
                # Some models may not support split_sentences kwarg.
                wav = self.tts.tts(text=text_clean, **kwargs)
        except TypeError:
            # Model doesn't support speaker/language; retry without extras
            try:
                wav = self.tts.tts(text=text_clean)
            except Exception:
                # Aggressive fallback: strip to alphanumerics and basic punctuation
                basic = re.sub(r"[^A-Za-z0-9 \.\,\!\?\;\:\'\-]", " ", text_clean)
                basic = re.sub(r"\s+", " ", basic).strip()
                if len(basic) < 8:
                    basic += " Ok."
                wav = self.tts.tts(text=basic)

        # Ensure float32 numpy and play with sounddevice
        import numpy as _np
        wav = _np.asarray(wav, dtype=_np.float32)
        if wav.ndim > 1:
            # Convert to mono if multi-channel
            wav = _np.mean(wav, axis=1).astype(_np.float32)
        
        # Apply speech rate control by adjusting playback sample rate
        # Lower sample rate = slower speech, higher sample rate = faster speech
        effective_sample_rate = int(self.sample_rate * self.speech_rate)
        sd.play(wav, effective_sample_rate)
        sd.wait()

# ---------------------------
# macOS TTS using built-in 'say' command
# ---------------------------
class MacOSTTS:
    def __init__(self, voice: str, speech_rate: float) -> None:
        """
        Uses macOS built-in 'say' command for text-to-speech.
        voice: Optional voice name (e.g., 'Alex', 'Samantha', 'Victoria')
        speech_rate: Speech rate in words per minute (default calculated from rate multiplier)
        """
        self.voice = voice
        # Convert speech_rate multiplier to words per minute
        # Normal speaking rate is around 175 WPM, adjust based on speech_rate
        base_wpm = 175
        self.words_per_minute = int(base_wpm * speech_rate)
        
    def speak(self, text: str):
        if not text:
            return
        
        # Clean text for better speech (remove markdown, etc.)
        text_clean = sanitize_text_for_tts(text, 1.0)  # Don't add extra pauses since 'say' handles rate
        
        # Build say command
        cmd = ["say"]
        if self.voice:
            cmd.extend(["-v", self.voice])
        cmd.extend(["-r", str(self.words_per_minute)])
        cmd.append(text_clean)
        
        try:
            # Run say command and wait for completion
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"[MacOS TTS] Error running 'say' command: {e}")
        except FileNotFoundError:
            print("[MacOS TTS] 'say' command not found. This feature requires macOS.")


# ---------------------------
# ElevenLabs TTS using SDK
# ---------------------------
class ElevenLabsTTS:
    def __init__(self, api_key: str, voice_id: str, voice_name: str, speech_rate: float, model: str, timeout: float) -> None:
        """
        Uses ElevenLabs SDK for text-to-speech.
        api_key: ElevenLabs API key (can also be set via ELEVENLABS_API_KEY env var)
        voice_id: Specific voice ID to use
        voice_name: Voice name to use (will look up ID automatically)
        speech_rate: Speech rate multiplier (implemented via stability/similarity settings)
        model: ElevenLabs model to use
        timeout: Timeout for ElevenLabs API requests
        """
        
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("ElevenLabs API key required. Set ELEVENLABS_API_KEY env var or pass api_key parameter.")
        
        self.client = ElevenLabs(api_key=self.api_key, timeout=timeout)
        self.voice_id = voice_id
        self.voice_name = voice_name
        self.speech_rate = speech_rate
        self.model = model
        self.timeout = timeout
        # If voice_name is provided but no voice_id, look it up
        if self.voice_name and not self.voice_id:
            self.voice_id = self._get_voice_id_by_name(self.voice_name)
        
    
    def _get_voice_id_by_name(self, name: str) -> str:
        """Look up voice ID by name."""
        try:
            voices = self.client.voices.get_all()
            
            for voice in voices.voices:
                if voice.name.lower() == name.lower():
                    print(f"Found ElevenLabs voice: {voice.name} ({voice.voice_id})")
                    return voice.voice_id
            
            print(f"Voice '{name}' not found. Available voices:")
            for voice in voices.voices[:10]:  # Show first 10 voices
                print(f"  - {voice.name} ({voice.voice_id})")
            
            raise ValueError(f"Voice '{name}' not found")
            
        except Exception as e:
            print(f"Error fetching voices: {e}")
            raise ValueError(f"Error fetching voices: {e}")
    
    def speak(self, text: str):
        if not text:
            return
        
        # Clean text for better speech
        text_clean = sanitize_text_for_tts(text, 1.0)  # Don't add extra pauses
        
        try:
            # Generate audio using the SDK
            audio = self.client.text_to_speech.convert(
                text=text_clean,
                voice_id=self.voice_id,
                model_id=self.model,
                output_format="mp3_44100_128",
            )
            
            # Play the audio directly
            play(audio)
                    
        except Exception as e:
            print(f"[ElevenLabs TTS] Error: {e}")


# ---------------------------
# Response editing utilities
# ---------------------------
def timeout_input(prompt: str, timeout: float) -> str:
    """
    Get input with a timeout. Returns empty string if timeout occurs.
    Works on Unix-like systems (macOS, Linux).
    """
    print(prompt, end='', flush=True)
    
    # Use select to check if input is available
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    
    if ready:
        return sys.stdin.readline().strip()
    else:
        print(f"\n(Timeout after {timeout}s - accepting as-is)")
        return ""

def edit_response(original_response: str, timeout: float) -> tuple[str, bool]:
    """
    Allow user to edit the model's response before it's spoken.
    Returns: (edited_response, should_speak)
    timeout: seconds to wait before auto-accepting (0 = no timeout)
    """
    print(f"\n--- Model Response ---")
    print(f"{original_response}")
    print(f"\n--- Edit Options ---")
    print("Press Enter to accept as-is")
    print("Type 'e' or 'edit' to edit the response")
    print("Type 's' or 'skip' to skip speaking (but keep in history)")
    print("Type 'r' or 'regenerate' to regenerate response (skip this one)")
    print("Type 'q' or 'quit' to exit")
    
    if timeout > 0:
        print(f"(Auto-accepting in {timeout}s if no input)")
    
    while True:
        if timeout > 0:
            choice = timeout_input("\nYour choice: ", timeout).lower()
            if choice == "":  # Timeout occurred
                return original_response, True
        else:
            choice = input("\nYour choice: ").strip().lower()
        
        if choice == "" or choice == "accept":
            return original_response, True
        elif choice in ["e", "edit"]:
            print(f"\nEdit the response (press Ctrl+C to cancel):")
            print(f"Original: {original_response}")
            try:
                edited = input("Edited:   ")
                if edited.strip():
                    return edited.strip(), True
                else:
                    print("Empty response, keeping original.")
                    return original_response, True
            except KeyboardInterrupt:
                print("\nEdit cancelled, keeping original.")
                return original_response, True
        elif choice in ["s", "skip"]:
            return original_response, False
        elif choice in ["r", "regenerate"]:
            return "", False  # Empty string signals regeneration needed
        elif choice in ["q", "quit"]:
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")

# ---------------------------
# Main loop
# ---------------------------
def main(args):
    # Handle special commands first
    if args.list_elevenlabs_voices:
        try:
            api_key = args.elevenlabs_api_key
            if not api_key:
                print("ERROR: ElevenLabs API key required. Set ELEVENLABS_API_KEY env var or use --elevenlabs-api-key")
                return
            client = ElevenLabs(api_key=api_key, timeout=args.timeout)
            voices = client.voices.get_all()

            print("Available ElevenLabs voices:")
            for voice in voices.voices:
                category = getattr(voice, 'category', 'Unknown')
                print(f"  - {voice.name} ({voice.voice_id}) - {category}")
        except Exception as e:
            print(f"ERROR: Failed to list ElevenLabs voices: {e}")
            return
        return

    # Create system prompt
    system_prompt_text = None
    if args.system_prompt_file:
        try:
            with open(args.system_prompt_file, "r", encoding="utf-8") as file:
                system_prompt_text = file.read()
        except Exception as e:
            print(f"[WARN] Could not read system prompt file: {e}")
            return

    if args.system_prompt:
        system_prompt_text += args.system_prompt

    client = LLMClient(base_url=args.llm_url, model=args.llm_model, timeout=args.timeout, system_prompt=system_prompt_text, max_history=args.max_history, api_key=args.llm_api_key)
    rec = VoiceRecorder(sample_rate=args.whisper_sample_rate, channels=args.whisper_channels, chunk_duration=args.whisper_chunk_duration, start_threshold=args.whisper_start_threshold, silence_timeout=args.whisper_silence_timeout, max_utterance=args.whisper_max_utterance)
    transcriber = WhisperTranscriber(model_name=args.whisper_model, device=args.whisper_device, language=args.whisper_language, sample_rate=args.whisper_sample_rate)
    
    # Choose TTS engine based on user preference
    if args.no_tts:
        tts = None
        print("TTS disabled")
    elif args.speech_engine == "coqui":
        tts = SimpleTTS(model_name=args.coqui_model, model_path=args.coqui_model_path, config_path=args.coqui_config_path, speaker=args.coqui_speaker, language=args.coqui_language, device=args.coqui_device, sample_rate=args.coqui_sample_rate, speech_rate=args.speech_rate)
        print("Using Coqui TTS")
    elif args.speech_engine == "elevenlabs":
        try:
            api_key = args.elevenlabs_api_key
            if not api_key:
                print("ERROR: ElevenLabs API key required. Set ELEVENLABS_API_KEY env var or use --elevenlabs-api-key")
                return
            voice = args.elevenlabs_voice
            # Determine if voice is ID or name
            voice_id = None
            voice_name = None

            if len(voice) == 20:  # ElevenLabs voice IDs are 20 characters
                voice_id = voice
            else:
                voice_name = voice
                
            tts = ElevenLabsTTS(api_key=api_key, voice_id=voice_id, voice_name=voice_name, speech_rate=args.speech_rate, model=args.elevenlabs_model, timeout=args.timeout)
            print("Using ElevenLabs TTS")
        except Exception as e:
            print(f"ERROR: Failed to initialize ElevenLabs TTS: {e}")
            return
    elif args.speech_engine == "macos":
        tts = MacOSTTS(voice=args.macos_voice, speech_rate=args.speech_rate)
        print("Using macOS 'say' command for TTS")
    else:
        print("ERROR: Invalid speech engine. Please use --speech-engine with a valid engine.")
        return

    print("\n--- Voice Chat Ready ---")
    print("Speak to start. Press Ctrl+C to exit.")
    print(f"Conversation history: {args.max_history} turns will be remembered")
    if args.edit_responses:
        if args.edit_timeout > 0:
            print(f"Response editing enabled - auto-accepting after {args.edit_timeout}s if no input")
        else:
            print("Response editing enabled - no timeout (manual input required)")

    try:
        while True:
            audio = rec.listen_once()
            if audio is None:
                continue
            print("Captured audio; transcribing...")
            try:
                text = transcriber.transcribe_numpy(audio)
            except Exception as e:
                print(f"Transcription error: {e}")
                continue
            if not text:
                print("(no transcription detected)")
                continue
            print(f"You: {text}")

            # Get response from LLM Server (with potential regeneration loop)
            final_reply = None
            first_attempt = True
            
            while True:
                try:
                    if first_attempt:
                        # First attempt: use normal chat (adds to history)
                        reply = client.chat(text)
                        first_attempt = False
                    else:
                        # Regeneration: don't add to history yet
                        reply = client.chat_without_history(text)
                except Exception as e:
                    print(f"[ERROR] LLM Server request failed: {e}")
                    break
                
                if args.edit_responses:
                    edit_timeout = args.edit_timeout
                    edited_reply, should_speak = edit_response(reply, edit_timeout)
                    if args.no_tts:
                        should_speak = False
                    if edited_reply == "":  # Regeneration requested
                        print("Regenerating response...")
                        continue
                    reply = edited_reply
                else:
                    should_speak = not args.no_tts
                
                final_reply = reply
                break  # Exit the regeneration loop
            
            if final_reply is None:
                continue  # Skip if no valid response was generated
                
            print(f"Model: {final_reply}")
            
            # If we regenerated, we need to add the final exchange to history
            if not first_attempt:
                client.add_to_history(text, final_reply)
            
            # Speak the reply if requested
            if should_speak:
                tts.speak(final_reply)

    except KeyboardInterrupt:
        print("\nExiting. Bye!")
        return

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice chat client for LLM Server (Whisper + Coqui/ElevenLabs/macOS TTS)")
    parser.add_argument("--timeout", "-t", type=float, default=REQUEST_TIMEOUT, help=f"Timeout for network requests in seconds (default: {REQUEST_TIMEOUT})")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS")
    parser.add_argument("--edit-responses", "-e",action="store_true", help="Allow editing model responses before they are spoken")
    parser.add_argument("--edit-timeout", type=float, default=EDIT_TIMEOUT, help=f"Seconds to wait for response editing input before auto-accepting. A value of 0 means no timeout and manual input is required. (default: {EDIT_TIMEOUT})")
    parser.add_argument("--llm-url", type=str, default=LLM_URL, help=f"Base URL of your LLM server (default: {LLM_URL})")
    parser.add_argument("--llm-model", type=str, default=LLM_MODEL, help=f"LLM model name to call on the server (default: {LLM_MODEL})")
    parser.add_argument("--llm-api-key", type=str, default=LLM_API_KEY, help=f"API key for the LLM server (default: {LLM_API_KEY})")
    parser.add_argument("--system-prompt", "-p", type=str, default=SYSTEM_PROMPT, help=f"Prompt to prepend as a system message (default: {SYSTEM_PROMPT})")
    parser.add_argument("--system-prompt-file", "-f", type=str, default=SYSTEM_PROMPT_FILE, help=f"Path to a UTF-8 text file containing the starting system prompt (default: {SYSTEM_PROMPT_FILE})")
    parser.add_argument("--max-history", type=int, default=MAX_HISTORY, help=f"Maximum number of conversation turns to keep in memory (default: {MAX_HISTORY})")
    parser.add_argument("--whisper-sample-rate", type=int, default=WHISPER_SAMPLE_RATE, help=f"Sample rate for Whisper transcription (default: {WHISPER_SAMPLE_RATE})")
    parser.add_argument("--whisper-channels", type=int, default=WHISPER_CHANNELS, help=f"Channels for Whisper transcription (default: {WHISPER_CHANNELS})")
    parser.add_argument("--whisper-chunk-duration", type=float, default=WHISPER_CHUNK_DURATION, help=f"Chunk duration for Whisper transcription (default: {WHISPER_CHUNK_DURATION})")
    parser.add_argument("--whisper-start-threshold", type=float, default=WHISPER_START_THRESHOLD, help=f"Start threshold for Whisper transcription (default: {WHISPER_START_THRESHOLD})")
    parser.add_argument("--whisper-silence-timeout", type=float, default=WHISPER_SILENCE_TIMEOUT, help=f"Silence timeout for Whisper transcription (default: {WHISPER_SILENCE_TIMEOUT})")
    parser.add_argument("--whisper-max-utterance", type=float, default=WHISPER_MAX_UTTERANCE_DURATION, help=f"Max utterance duration for Whisper transcription (default: {WHISPER_MAX_UTTERANCE_DURATION})")
    parser.add_argument("--whisper-model", default=WHISPER_MODEL, choices=["tiny", "base", "small", "medium", "large"], help=f"Whisper model name (default: {WHISPER_MODEL})")
    parser.add_argument("--whisper-device", type=str, default=WHISPER_DEVICE, help=f"Device for whisper (default: {WHISPER_DEVICE})")
    parser.add_argument("--whisper-language", type=str, default=WHISPER_LANGUAGE, help=f"Language hint for whisper, e.g. 'en' (default: {WHISPER_LANGUAGE})")
    parser.add_argument("--speech-engine", "-s", type=str, default=SPEECH_ENGINE, choices=["coqui", "elevenlabs", "macos"], help=f"Speech engine to use (default: {SPEECH_ENGINE})")
    parser.add_argument("--speech-rate", "-r", type=float, default=SPEECH_RATE, help=f"Speech rate control: 1.0 = normal, 0.5 = half speed (slower), 2.0 = double speed (faster) (default: {SPEECH_RATE})")
    parser.add_argument("--coqui-model", default=COQUI_MODEL, help="Coqui TTS model name (e.g., tts_models/en/ljspeech/tacotron2-DDC)")
    parser.add_argument("--coqui-model-path", default=COQUI_MODEL_PATH, help="Coqui TTS model path (e.g., tts_models/en/ljspeech/tacotron2-DDC)")
    parser.add_argument("--coqui-config-path", default=COQUI_CONFIG_PATH, help="Coqui TTS config path (e.g., config.json)")
    parser.add_argument("--coqui-speaker", default=COQUI_SPEAKER, help="Optional speaker name/id for multispeaker models")
    parser.add_argument("--coqui-language", default=None, help="Optional language code/name for multilingual models")
    parser.add_argument("--coqui-device", default=COQUI_DEVICE, help="Device for Coqui TTS: 'cpu' or 'cuda'")
    parser.add_argument("--coqui-sample-rate", type=int, default=COQUI_SAMPLE_RATE, help=f"Sample rate for Coqui TTS (default: {COQUI_SAMPLE_RATE})")
    parser.add_argument("--list-elevenlabs-voices", "-l", action="store_true", help="List available ElevenLabs voices and exit")
    parser.add_argument("--elevenlabs-api-key", default=ELEVENLABS_API_KEY, help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)")
    parser.add_argument("--elevenlabs-model", default=ELEVENLABS_MODEL, help="ElevenLabs model name (default: {ELEVENLABS_MODEL})")
    parser.add_argument("--elevenlabs-voice", default=ELEVENLABS_VOICE, help="ElevenLabs voice name or ID (e.g., 'Rachel', 'Adam', or voice ID)")
    parser.add_argument("--macos-voice", default=MACOS_VOICE, help=f"macOS voice name (e.g., 'Alex', 'Samantha', 'Victoria') (default: {MACOS_VOICE})")
    
    args = parser.parse_args()
    main(args)
