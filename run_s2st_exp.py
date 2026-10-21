#!/usr/bin/env python3
"""Re-run notebook pipeline via function calls using the cloned s2st-fairseq repo."""
import os
import sys
import glob
import urllib.request
import argparse
from pathlib import Path

FAIRSEQ_ROOT = os.path.expanduser("~/s2st-fairseq-exp/fairseq")
os.chdir(FAIRSEQ_ROOT)
sys.path.insert(0, FAIRSEQ_ROOT)

import numpy as np
import soundfile as sf

# ============ STEP 2: download pretrained models ============
def download_file(url, output_path):
    if os.path.exists(output_path):
        print(f"exists: {output_path}")
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"downloading {url} -> {output_path}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"done: {output_path}")

MODEL_URLS = {
    "pretrained_models/vocoder/g_00500000": "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/g_00500000",
    "pretrained_models/vocoder/config.json": "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/config.json",
    "pretrained_models/hubert/hubert_base_ls960.pt": "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt",
    "pretrained_models/hubert/hubert_base_ls960_L6_km100.bin": "https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/hubert_base_ls960_L6_km100.bin",
}
for tp, url in MODEL_URLS.items():
    download_file(url, tp)

VOCODER_CKPT = "pretrained_models/vocoder/g_00500000"
VOCODER_CFG = "pretrained_models/vocoder/config.json"
HUBERT_CKPT = "pretrained_models/hubert/hubert_base_ls960.pt"
HUBERT_KM = "pretrained_models/hubert/hubert_base_ls960_L6_km100.bin"

# ============ STEP 3: dummy dataset ============
def create_dummy_dataset(base_dir="toy_dataset", splits=("train", "dev"), num_samples=8, sr=16000):
    src_dir = os.path.join(base_dir, "src_audio")
    tgt_dir = os.path.join(base_dir, "tgt_audio")
    for split in splits:
        os.makedirs(os.path.join(src_dir, split), exist_ok=True)
        os.makedirs(os.path.join(tgt_dir, split), exist_ok=True)
        for idx in range(num_samples):
            freq = 300 + idx * 50
            audio = np.sin(2 * np.pi * freq * np.linspace(0, 2, sr * 2)).astype(np.float32)
            sf.write(os.path.join(src_dir, split, f"sample_{idx}.wav"), audio, sr)
            sf.write(os.path.join(tgt_dir, split, f"sample_{idx}.wav"), audio, sr)
    print(f"dummy dataset at {base_dir}")
    return src_dir, tgt_dir

DATA_DIR = "toy_dataset"
SRC_AUDIO, TGT_AUDIO = create_dummy_dataset(DATA_DIR)
DATA_ROOT = os.path.join(DATA_DIR, "s2ut_data")

# ============ STEP 3.1: unit extraction (FIXED: manifest file, not directory) ============
def create_wav_manifest(audio_dir, manifest_path):
    """quantize_with_kmeans expects a MANIFEST FILE: line1=root, then relpath\\tframes."""
    wav_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    with open(manifest_path, "w") as f:
        f.write(os.path.abspath(audio_dir) + "\n")
        for w in wav_files:
            info = sf.info(w)
            rel = os.path.relpath(w, audio_dir)
            f.write(f"{rel}\t{info.frames}\n")
    print(f"manifest: {manifest_path} ({len(wav_files)} files)")
    return manifest_path

from examples.textless_nlp.gslm.speech2unit.clustering.quantize_with_kmeans import (
    get_parser as get_quantize_parser,
    main as quantize_main,
    get_logger,
)

def extract_target_units(
    audio_dir,
    output_file,
    kmeans_model,
    acoustic_model,
    layer=6,
    feature_type="hubert",
    extension=".wav",
):
    manifest_path = audio_dir.rstrip("/") + "_manifest.tsv"
    create_wav_manifest(audio_dir, manifest_path)
    parser = get_quantize_parser()
    args = parser.parse_args(
        [
            "--feature_type", feature_type,
            "--kmeans_model_path", str(kmeans_model),
            "--acoustic_model_path", str(acoustic_model),
            "--layer", str(layer),
            "--manifest_path", str(manifest_path),
            "--out_quantized_file_path", str(output_file),
            "--extension", extension,
        ]
    )
    quantize_main(args, get_logger())
    print(f"units -> {output_file}")

for split in ["train", "dev"]:
    extract_target_units(
        audio_dir=os.path.join(TGT_AUDIO, split),
        output_file=os.path.join(TGT_AUDIO, f"{split}.txt"),
        kmeans_model=HUBERT_KM,
        acoustic_model=HUBERT_CKPT,
        layer=6,
    )

with open(os.path.join(TGT_AUDIO, "train.txt")) as f:
    print("sample units line:", f.readline()[:120])

# ============ STEP 3.2: prep S2UT data ============
from examples.speech_to_speech.preprocessing.prep_s2ut_data import process as prep_s2ut_process

def prepare_s2ut_dataset(
    source_dir,
    target_dir,
    output_root,
    splits=("train", "dev"),
    reduce_unit=True,
    vocoder_ckpt=None,
    vocoder_cfg=None,
):
    args = argparse.Namespace(
        source_dir=Path(source_dir),
        target_dir=Path(target_dir),
        data_split=list(splits),
        output_root=Path(output_root),
        reduce_unit=reduce_unit,
        vocoder_checkpoint=str(vocoder_ckpt) if vocoder_ckpt else None,
        vocoder_cfg=str(vocoder_cfg) if vocoder_cfg else None,
    )
    prep_s2ut_process(args)
    print(f"S2UT data -> {output_root}")

prepare_s2ut_dataset(SRC_AUDIO, TGT_AUDIO, DATA_ROOT, ("train", "dev"), True, VOCODER_CKPT, VOCODER_CFG)
print("--- train.tsv head ---")
with open(os.path.join(DATA_ROOT, "train.tsv")) as f:
    for i, line in enumerate(f):
        print(line[:160].rstrip())
        if i >= 2:
            break
print("config.yaml exists:", os.path.exists(os.path.join(DATA_ROOT, "config.yaml")))

# ============ STEP 4: TRAIN (CPU, via fairseq_cli.train.main) ============
MODEL_DIR = "checkpoints/s2ut_model"
os.makedirs(MODEL_DIR, exist_ok=True)

from fairseq import options
from fairseq_cli import train as fairseq_train

def build_train_argv(data_root, model_dir, max_update=30):
    return [
        data_root,
        "--config-yaml", "config.yaml",
        "--task", "speech_to_speech",
        "--target-is-code", "--target-code-size", "100", "--vocoder", "code_hifigan",
        "--criterion", "speech_to_unit", "--label-smoothing", "0.2",
        "--arch", "s2ut_transformer_fisher", "--share-decoder-input-output-embed",
        "--dropout", "0.1", "--attention-dropout", "0.1", "--relu-dropout", "0.1",
        "--train-subset", "train", "--valid-subset", "dev",
        "--save-dir", model_dir,
        "--lr", "0.0005", "--lr-scheduler", "inverse_sqrt",
        "--warmup-init-lr", "1e-7", "--warmup-updates", "10",
        "--optimizer", "adam", "--adam-betas", "(0.9,0.98)", "--clip-norm", "10.0",
        "--max-update", str(max_update),
        "--max-tokens", "4000", "--max-target-positions", "3000",
        "--update-freq", "1", "--seed", "1",
        "--num-workers", "0",
        "--required-batch-size-multiple", "1",
        "--log-format", "simple", "--log-interval", "1",
    ]

print("=== TRAINING (30 updates, CPU) ===")
train_parser = options.get_training_parser()
train_args = options.parse_args_and_arch(train_parser, build_train_argv(DATA_ROOT, MODEL_DIR, 30))
try:
    fairseq_train.main(train_args)
    print("training finished OK")
except SystemExit as e:
    print("train SystemExit:", e)

ckpts = sorted(glob.glob(os.path.join(MODEL_DIR, "checkpoint*.pt")))
print("checkpoints:", ckpts)

# ============ STEP 5: GENERATE + VOICE ============
RESULTS_PATH = "results/s2ut_results"
GEN_SUBSET = "dev"
CHECKPOINT_PATH = ckpts[-1] if ckpts else None
# prefer checkpoint_best if present
best = os.path.join(MODEL_DIR, "checkpoint_best.pt")
if os.path.exists(best):
    CHECKPOINT_PATH = best

if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
    print("NO CHECKPOINT - stop")
    sys.exit(1)

from fairseq import options as fairseq_options
from fairseq_cli import generate as fairseq_generate

def generate_discrete_units(data_root, checkpoint_path, results_path, gen_subset="dev", beam=10, max_tokens=20000):
    os.makedirs(results_path, exist_ok=True)
    argv = [
        data_root,
        "--config-yaml", "config.yaml",
        "--task", "speech_to_speech",
        "--target-is-code", "--target-code-size", "100", "--vocoder", "code_hifigan",
        "--path", str(checkpoint_path),
        "--gen-subset", str(gen_subset),
        "--max-tokens", str(max_tokens),
        "--beam", str(beam),
        "--max-len-a", "1",
        "--results-path", str(results_path),
    ]
    gen_parser = fairseq_options.get_generation_parser()
    gen_args = fairseq_options.parse_args_and_arch(gen_parser, argv)
    fairseq_generate.main(gen_args)
    print(f"generate log -> {results_path}")

def extract_units_from_log(log_path, output_unit_path):
    hypotheses = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("D-"):
                parts = line.strip().split("\t")
                sample_id = int(parts[0].split("-")[1])
                tokens = parts[2] if len(parts) > 2 else ""
                hypotheses.append((sample_id, tokens))
    hypotheses.sort(key=lambda x: x[0])
    with open(output_unit_path, "w", encoding="utf-8") as f:
        for _, tokens in hypotheses:
            f.write(f"{tokens}\n")
    print(f"extracted {len(hypotheses)} unit sequences -> {output_unit_path}")

from examples.speech_to_speech.generate_waveform_from_code import main as generate_waveform_main

def synthesize_waveform(in_code_file, vocoder_ckpt, vocoder_cfg, results_path,
                        dur_prediction=True, speaker_id=-1, cpu=True):
    args = argparse.Namespace(
        in_code_file=str(in_code_file),
        vocoder=str(vocoder_ckpt),
        vocoder_cfg=str(vocoder_cfg),
        results_path=str(results_path),
        dur_prediction=dur_prediction,
        speaker_id=speaker_id,
        cpu=cpu,
    )
    generate_waveform_main(args)
    print(f"waveforms -> {results_path}")

print("=== GENERATION ===")
generate_discrete_units(DATA_ROOT, CHECKPOINT_PATH, RESULTS_PATH, gen_subset=GEN_SUBSET)
log_file = f"{RESULTS_PATH}/generate-{GEN_SUBSET}.txt"
unit_file = f"{RESULTS_PATH}/generate-{GEN_SUBSET}.unit"
extract_units_from_log(log_file, unit_file)
print("--- unit file ---")
with open(unit_file) as f:
    print(f.read())

print("=== VOCODING ===")
synthesize_waveform(unit_file, VOCODER_CKPT, VOCODER_CFG, RESULTS_PATH, dur_prediction=True, cpu=True)

wavs = sorted(glob.glob(f"{RESULTS_PATH}/*.wav"))
print("generated wavs:", wavs)
for w in wavs:
    info = sf.info(w)
    print(f"  {w}: sr={info.samplerate}, frames={info.frames}, dur={info.duration:.2f}s")

print("PIPELINE COMPLETE")
