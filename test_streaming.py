from datasets import load_dataset

print("Loading LibriSpeech with streaming...")
dataset = load_dataset(
    "librispeech_asr",
    "clean",
    split="train.100",
    streaming=True,
)

print("Taking first sample...")
first_sample = next(iter(dataset))
print(f"First sample keys: {first_sample.keys()}")
print(f"Text: {first_sample.get('text', 'NO TEXT')[:50]}...")
print(f"Audio shape: {len(first_sample['audio']['array']) if 'audio' in first_sample else 'NO AUDIO'}")
print("Success!")
