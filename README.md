Tested on WSL2 5090 only.

This is a very rough script finetune script I vibecoded. There is a dataprep script, the idea is to drag audio files into a folder, then the script preps the dataset/metadata. Then finetune script.

Install echo-tts as normal.


1 . python finetune_cli.py prepare \
  --raw_dir "/mnt/h/XX/XX/New folder/New folder" \
  --segments_dir "datasets/XX/ar/segments" \
  --max_duration 30


2. python finetune_cli.py transcribe \
  --segments_dir "datasets/XX/ar/segments" \
  --metadata "datasets/XX/ar/metadata.jsonl" \
  --lang ar



3. python finetune_cli.py train \
  --resume "checkpoints/echo_arabic_base/echo_abu_step16000.pt" \
  --segments_dir "Arabic_Base/segments" \
  --metadata "Arabic_Base/metadata.jsonl" \
  --out_dir "checkpoints/echo_arabic_base" \
  --lang ar \
  --batch_size 2 \
  --steps 20000 \
  --lr 2e-5 \
  --device cuda \

Inference :
4. python infer_cli.py \
  --checkpoint checkpoints/echo_abu_ar/echo_abu_final.pt \
  --speaker "/mnt/h/Audio/AbuTala/New folder/abu30sec.mp3" \
  --text "[S1] في يوم من الأيام، شاب فقير قرر يتحدى العالم، خسر كثير، لكنه تعلّم أكثر، واليوم صوته يحمل وجع الماضي وقوة الإصرار." \
  --out outputs/abu_ar_final_test.wav \
  --device cuda \
  --num_steps 40 \
  --cfg_text 7.0 \
  --cfg_speaker 1.1 \

In this example I trained arabic, but English is possible of course. And theory a different language. 

I am not an expert. This could be a bad way of doing things, test, improve etc. As such I probably can't help with any techical queries. 



# Echo-TTS

A multi-speaker text-to-speech model with speaker reference conditioning. See the [blog post](https://jordandarefsky.com/blog/2025/echo/) for technical details.

**Model:** [jordand/echo-tts-base](https://huggingface.co/jordand/echo-tts-base) | **Demo:** [echo-tts-preview](https://huggingface.co/spaces/jordand/echo-tts-preview)

This work was made possible by the TPU Research Cloud (TRC).

## Responsible Use

Don't use this model to:
- Impersonate real people without their consent
- Generate deceptive audio (e.g., fraud, misinformation, deepfakes)

You are responsible for complying with local laws regarding biometric data and voice cloning.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU with at least 8GB VRAM.

## Quick Start

### Gradio UI

```bash
python gradio_app.py
```

### Python API

```python
from inference import (
    load_model_from_hf,
    load_fish_ae_from_hf,
    load_pca_state_from_hf,
    load_audio,
    sample_pipeline,
    sample_euler_cfg_independent_guidances,
)
from functools import partial
import torchaudio

# Load models (downloads from HuggingFace on first run)
model = load_model_from_hf(delete_blockwise_modules=True)
fish_ae = load_fish_ae_from_hf()
pca_state = load_pca_state_from_hf()

# Load speaker reference (or set to None for no reference)
speaker_audio = load_audio("speaker.wav").cuda()

# Configure sampler
sample_fn = partial(
    sample_euler_cfg_independent_guidances,
    num_steps=40,
    cfg_scale_text=3.0,
    cfg_scale_speaker=8.0,
    cfg_min_t=0.5,
    cfg_max_t=1.0,
    truncation_factor=None,
    rescale_k=None,
    rescale_sigma=None,
    speaker_kv_scale=None,
    speaker_kv_max_layers=None,
    speaker_kv_min_t=None,
    sequence_length=640, # (~30 seconds)
)

# Generate
text = "[S1] Hello, this is a test of the Echo TTS model."
audio_out, _ = sample_pipeline(
    model=model,
    fish_ae=fish_ae,
    pca_state=pca_state,
    sample_fn=sample_fn,
    text_prompt=text,
    speaker_audio=speaker_audio,
    rng_seed=0,
)

torchaudio.save("output.wav", audio_out[0].cpu(), 44100)
```

See also:
- `inference.py` -- lower-level usage example at the bottom of the file
- `inference_blockwise.py` -- examples of blockwise/continuation generation

## Low VRAM (8GB)

In `gradio_app.py`, adjust:

```python
FISH_AE_DTYPE = torch.bfloat16  # instead of float32
DEFAULT_SAMPLE_LATENT_LENGTH = 576  # (< 640 depending on what fits) instead of 640
```

## Tips

### Generation Length

Echo is trained to generate up to 30 seconds of audio (640 latents) given text and reference audio. Since the supplied text always corresponded to ≤30 seconds of audio during training, the model will attempt to fit any text prompt at inference into the 30 seconds of generated audio (and thus, e.g., long text prompts may result in faster speaking rates). On the other hand, shorter text prompts will work and will produce shorter outputs (as the model generates latent padding automatically).

If "Sample Latent Length" (in Custom Shapes in gradio)/sequence_length is set to less than 640, the model will attempt to generate the prefix corresponding to that length. I.e., if you set this to 320, and supply ~30 seconds worth of text, the model will likely generate the first half of the text (rather than try to fit the entirety of the text into the first 15 seconds).

### Reference Audio

You can condition on up to 5 minutes of reference audio, but shorter clips (e.g., 10 seconds or shorter) work well too.

### Force Speaker (KV Scaling)

Sometimes out-of-distribution text for a given reference speaker will cause the model to generate a different speaker entirely. Enabling "Force Speaker" (which scales speaker KV for a portion of timesteps, default scale 1.5) generally fixes this. However, high values may introduce artifacts or "overconditioning." Aim for the lowest scale that produces the correct speaker: 1.0 is baseline, 1.5 is the default when enabled and will usually force the speaker, but lower values (e.g., 1.3, 1.1) may suffice.

### Text Prompt Format

Text prompts use the format from [WhisperD](https://huggingface.co/jordand/whisper-d-v1a). Colons, semicolons, and emdashes are normalized to commas (see inference.py tokenizer_encode) by default, and "[S1] " will be added to the beginning of the prompt if not already present. Commas generally function as pauses. Exclamation points (and other non-bland punctuation) may lead to increased expressiveness but also potentially lower quality on occasion; improving controllability is an important direction for future work.

The included text presets are stylistically in-distribution with the WhisperD transcription style.

### Blockwise Generation

`inference_blockwise.py` includes blockwise sampling, which allows generating audio in smaller blocks as well as producing continuations of existing audio (where the prefix and continuation are up to 30 seconds combined). The model released on HF is a fully fine-tuned model (not the LoRA as described in the blog). Blockwise generation enables audio streaming (not included in current code) since the S1-DAC decoder is causal. Blockwise functionality hasn't been thoroughly tested and may benefit from different (e.g., smaller) CFG scales.

## License

Code in this repo is MIT‑licensed except where file headers specify otherwise (e.g., autoencoder.py is Apache‑2.0).

Regardless of our model license, audio outputs are CC-BY-NC-SA-4.0 due to the dependency on the Fish Speech S1-DAC autoencoder, which is CC-BY-NC-SA-4.0.

We have chosen to release the Echo-TTS weights under CC-BY-NC-SA-4.0.

For included audio prompts, see `audio_prompts/LICENSE`.

## Citation

```bibtex
@misc{darefsky2025echo,
    author = {Darefsky, Jordan},
    title = {Echo-TTS},
    year = {2025},
    url = {https://jordandarefsky.com/blog/2025/echo/}
}
```
