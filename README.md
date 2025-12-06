# LLM Voice Chat

A fully-featured voice chat client for interacting with LLM servers using speech. Talk naturally and get spoken responses back!

## Features

- **Voice Activity Detection (VAD)**: Automatic detection of when you start and stop speaking using RMS threshold
- **Local Transcription**: Uses OpenAI's Whisper for speech-to-text conversion
- **LLM Integration**: Connects to OpenAI-compatible API endpoints (LM Studio, LocalAI, etc.)
- **Multiple TTS Engines**: Choose from:
  - **Coqui TTS** - Fully local, open-source text-to-speech
  - **ElevenLabs** - High-quality cloud-based TTS
  - **macOS Say** - Built-in macOS text-to-speech
- **Conversation History**: Maintains context across multiple turns
- **Response Editing**: Optional feature to review and edit responses before they're spoken
- **Customizable System Prompts**: Configure the AI's personality and behavior
- **Speech Rate Control**: Adjust speaking speed (0.5x to 2x)

## Requirements

- **Python 3.11** (required)
- Microphone for voice input
- LLM server with OpenAI-compatible API endpoint (e.g., LM Studio, Ollama, LocalAI)
- Audio output device for TTS playback

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sacredbanana/LLMVoiceChat
cd LLMVoiceChat
```

### 2. Create a Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Whisper Model

The first time you run the application, Whisper will automatically download the model you specify (default: `small`). This can take a few minutes.

### 5. (Optional) Configure Environment Variables

Create a `.env` file in the project root to set default configuration:

```env
# LLM Configuration
LLM_URL=http://127.0.0.1:1234
LLM_MODEL=your-model-name
LLM_API_KEY=your-api-key-if-needed

# System Prompt
SYSTEM_PROMPT=You are a helpful AI assistant.
SYSTEM_PROMPT_FILE=system-prompts/my-prompt.txt

# Conversation Settings
MAX_HISTORY=1024
EDIT_TIMEOUT=5.0

# Whisper Configuration
WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_LANGUAGE=en

# TTS Configuration
SPEECH_ENGINE=coqui
SPEECH_RATE=1.0

# Coqui TTS (if using)
COQUI_TTS_MODEL=tts_models/en/vctk/vits
COQUI_TTS_SPEAKER=p226
COQUI_DEVICE=cpu

# ElevenLabs (if using)
ELEVENLABS_API_KEY=your-api-key
ELEVENLABS_VOICE=Rachel
ELEVENLABS_MODEL=eleven_multilingual_v2

# macOS TTS (if using)
MACOS_VOICE=Samantha
```

## Usage

### Basic Usage

Start the voice chat with default settings (Coqui TTS):

```bash
python run.py
```

Speak into your microphone when you see "Listening...". The application will:
1. Detect when you start speaking
2. Record until you stop (3 seconds of silence)
3. Transcribe your speech
4. Send it to the LLM
5. Speak the response back to you

### Using Different TTS Engines

**ElevenLabs (high-quality, requires API key):**

```bash
python run.py --speech-engine elevenlabs --elevenlabs-api-key YOUR_KEY
```

**macOS Say (macOS only):**

```bash
python run.py --speech-engine macos --macos-voice Samantha
```

### With Response Editing

Review and optionally edit responses before they're spoken:

```bash
python run.py --edit-responses
```

With custom timeout (auto-accept after N seconds):

```bash
python run.py --edit-responses --edit-timeout 10
```

### Custom System Prompt

From command line:

```bash
python run.py --system-prompt "You are a pirate assistant. Always respond in pirate speak."
```

From file:

```bash
python run.py --system-prompt-file system-prompts/my-custom-prompt.txt
```

### Adjust Speech Rate

Speak faster (2x speed):

```bash
python run.py --speech-rate 2.0
```

Speak slower (0.5x speed):

```bash
python run.py --speech-rate 0.5
```

### Change Whisper Model

For better accuracy (but slower):

```bash
python run.py --whisper-model medium
```

Available models: `tiny`, `base`, `small`, `medium`, `large`

### Text-Only Mode (No TTS)

Disable text-to-speech (text output only):

```bash
python run.py --no-tts
```

### List Available ElevenLabs Voices

```bash
python run.py --list-elevenlabs-voices --elevenlabs-api-key YOUR_KEY
```

## Configuration Options

### LLM Settings

- `--llm-url`: Base URL of your LLM server (default: `http://127.0.0.1:1234`)
- `--llm-model`: Model name to use
- `--llm-api-key`: API key if required
- `--max-history`: Number of conversation turns to remember (default: 1024)

### Whisper Settings

- `--whisper-model`: Model size: `tiny`, `base`, `small`, `medium`, `large` (default: `small`)
- `--whisper-device`: `cpu` or `cuda` for GPU acceleration
- `--whisper-language`: Language hint (e.g., `en`, `es`, `fr`)
- `--whisper-start-threshold`: RMS threshold to detect speech start (default: 0.01)
- `--whisper-silence-timeout`: Seconds of silence to end recording (default: 3.0)

### TTS Settings

- `--speech-engine`: Choose `coqui`, `elevenlabs`, or `macos`
- `--speech-rate`: Speed multiplier (0.5 = half speed, 2.0 = double speed)
- `--no-tts`: Disable text-to-speech entirely

#### Coqui TTS Options

- `--coqui-model`: TTS model name (default: `tts_models/en/vctk/vits`)
- `--coqui-speaker`: Speaker ID for multi-speaker models
- `--coqui-device`: `cpu` or `cuda`

#### ElevenLabs Options

- `--elevenlabs-api-key`: Your ElevenLabs API key
- `--elevenlabs-voice`: Voice name or ID
- `--elevenlabs-model`: Model to use (default: `eleven_multilingual_v2`)

#### macOS Options

- `--macos-voice`: Voice name (e.g., `Alex`, `Samantha`, `Victoria`)

### Response Editing

- `--edit-responses` or `-e`: Enable response editing mode
- `--edit-timeout`: Seconds to wait before auto-accepting (0 = manual only)

## Troubleshooting

### Microphone Not Working

Check your system's audio input settings and ensure the microphone is set as the default input device.

### Whisper Model Download Issues

If the Whisper model fails to download, check your internet connection and ensure you have sufficient disk space (~500MB for small model).

### LLM Connection Errors

- Verify your LLM server is running
- Check the `--llm-url` matches your server's address
- Ensure the server is configured for OpenAI-compatible API endpoints

### TTS Issues

**Coqui TTS:**
- First run may take time to download models
- Try switching to a different model if you encounter errors

**ElevenLabs:**
- Verify your API key is valid
- Check your API quota/credits

**macOS:**
- Only works on macOS
- Try `say -v "?"` in terminal to list available voices

### Voice Detection Too Sensitive/Not Sensitive Enough

Adjust the threshold:

```bash
# More sensitive (lower threshold)
python run.py --whisper-start-threshold 0.005

# Less sensitive (higher threshold)
python run.py --whisper-start-threshold 0.02
```

### Response Cut Off Too Early/Late

Adjust the silence timeout:

```bash
# Wait longer before stopping (5 seconds)
python run.py --whisper-silence-timeout 5.0

# Stop sooner (1.5 seconds)
python run.py --whisper-silence-timeout 1.5
```

## Advanced Features

### GPU Acceleration

For faster Whisper transcription with CUDA-capable GPU:

```bash
python run.py --whisper-device cuda
```

For Coqui TTS with GPU:

```bash
python run.py --coqui-device cuda
```

### Long Conversations

Increase conversation history limit:

```bash
python run.py --max-history 2048
```

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) - Speech recognition
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Text-to-speech synthesis
- [ElevenLabs](https://elevenlabs.io/) - High-quality TTS API
