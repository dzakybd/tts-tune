import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

AUDIO_PROMPT_PATH = "ono.wav"
text = "Halo apa kabar? saya ono, senang berkenalan denganmu."
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH,)
ta.save("output-ono.wav", wav, model.sr)