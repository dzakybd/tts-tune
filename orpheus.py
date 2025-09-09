from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/orpheus-3b-0.1-ft",
    max_seq_length= 2048, # Choose any for long context!
    dtype = None, # Select None for auto detection
    load_in_4bit = False, # Select True for 4bit which reduces memory usage
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

import pandas as pd
from datasets import Dataset, Audio
from pathlib import Path

base_dir = Path("output")

# Load CSV with pandas
df = pd.read_csv(base_dir / "metadata.csv")
df = df[df['duration']<=30]

# Fix paths and rename columns in pandas
df["audio"] = df["filepath"].apply(lambda x: str(base_dir / x))
df = df.rename(columns={"transcript": "text"}).drop(columns=["filepath"])

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Cast audio column
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

print(dataset)
print(dataset[0]["audio"])

#@title Tokenization Function

import locale
import torchaudio.transforms as T
import os
import torch
from snac import SNAC
locale.getpreferredencoding = lambda: "UTF-8"
ds_sample_rate = dataset[0]["audio"]["sampling_rate"]

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to("cuda")
def tokenise_audio(waveform):
  waveform = torch.from_numpy(waveform).unsqueeze(0)
  waveform = waveform.to(dtype=torch.float32)
  resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
  waveform = resample_transform(waveform)

  waveform = waveform.unsqueeze(0).to("cuda")

  #generate the codes from snac
  with torch.inference_mode():
    codes = snac_model.encode(waveform)

  all_codes = []
  for i in range(codes[0].shape[1]):
    all_codes.append(codes[0][0][i].item()+128266)
    all_codes.append(codes[1][0][2*i].item()+128266+4096)
    all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
    all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
    all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
    all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
    all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))


  return all_codes

def add_codes(example):
    # Always initialize codes_list to None
    codes_list = None

    try:
        answer_audio = example.get("audio")
        # If there's a valid audio array, tokenise it
        if answer_audio and "array" in answer_audio:
            audio_array = answer_audio["array"]
            codes_list = tokenise_audio(audio_array)
    except Exception as e:
        print(f"Skipping row due to error: {e}")
        # Keep codes_list as None if we fail
    example["codes_list"] = codes_list

    return example

dataset = dataset.map(add_codes, remove_columns=["audio"])

tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009

start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai =  tokeniser_length + 6
pad_token = tokeniser_length + 7

audio_tokens_start = tokeniser_length + 10

dataset = dataset.filter(lambda x: x["codes_list"] is not None)
dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0)

def remove_duplicate_frames(example):
    vals = example["codes_list"]
    if len(vals) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = vals[:7]

    removed_frames = 0

    for i in range(7, len(vals), 7):
        current_first = vals[i]
        previous_first = result[-7]

        if current_first != previous_first:
            result.extend(vals[i:i+7])
        else:
            removed_frames += 1

    example["codes_list"] = result

    return example

dataset = dataset.map(remove_duplicate_frames)

tok_info = '''*** HERE you can modify the text prompt
If you are training a multi-speaker model (e.g., canopylabs/orpheus-3b-0.1-ft),
ensure that the dataset includes a "source" field and format the input accordingly:
- Single-speaker: f"{example['text']}"
- Multi-speaker: f"{example['source']}: {example['text']}"
'''
print(tok_info)

def create_input_ids(example):
    # Determine whether to include the source field
    text_prompt = f"{example['source']}: {example['text']}" if "source" in example else example["text"]

    text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
    text_ids.append(end_of_text)

    example["text_tokens"] = text_ids
    input_ids = (
        [start_of_human]
        + example["text_tokens"]
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + example["codes_list"]
        + [end_of_speech]
        + [end_of_ai]
    )
    example["input_ids"] = input_ids
    example["labels"] = input_ids
    example["attention_mask"] = [1] * len(input_ids)

    return example


dataset = dataset.map(create_input_ids, remove_columns=["text", "codes_list"])
columns_to_keep = ["input_ids", "labels", "attention_mask"]
columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]

dataset = dataset.remove_columns(columns_to_remove)

from transformers import TrainingArguments,Trainer,DataCollatorForSeq2Seq
trainer = Trainer(
    model = model,
    train_dataset = dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
        remove_unused_columns=False
    ),
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")

model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
