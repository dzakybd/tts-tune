import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
# if torch.cuda.is_available():
#     device = "cuda"
# elif torch.backends.mps.is_available():
#     device = "mps"
# else:
device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_local(ckpt_dir="checkpoints/chatterbox_finetuned", device=device)
# model = ChatterboxTTS.from_pretrained(device=device)

AUDIO_PROMPT_PATH = "ono.wav"
text = "Detektif sempat membantah tuduhan sepihak itu. Namun setelah ditelusuri, ternyata saat ini sang manajer sangat sulit untuk dihubungi dan keberadaannya pun tidaklah diketahui."

# Higher exaggeration tends to speed up speech (0.25 - 2)
#  cfg_weight helps compensate with slower, more deliberate pacing (0.0 - 1)
wav = model.generate(
    text,
    audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
)
ta.save("output-ono.wav", wav, model.sr)