#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube -> WAV (24k mono) -> faster-whisper (segments only)
-> Alignment = pass-through (romanize Arabic text only)
-> Split/export per-video folder + [video_id]_transcripts.txt (filename,duration,transcript)
-> Summary computed from all transcript CSVs (no audio decode)

Output under ./output:
- output/downloaded/[id]_[title].wav
- output/transcription/[id]_[title].json
- output/alignment/[id]_[title].json           (same segments, Arabic romanized)
- output/splitted/[id]_[title]/[id]_[NNNN].wav
- output/splitted/[id]_[title]/[id]_transcripts.txt  (CSV: filename,duration,transcript)
- output/summary.json
"""

import re
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import yt_dlp
from pydub import AudioSegment
from faster_whisper import WhisperModel


# --------------------------- paths ---------------------------

BASE_DIR = Path("output").resolve()
DIR_DOWN  = BASE_DIR / "downloaded"
DIR_TRANS = BASE_DIR / "transcription"
DIR_ALIGN = BASE_DIR / "alignment"
DIR_SPLIT = BASE_DIR / "splitted"
for d in (BASE_DIR, DIR_DOWN, DIR_TRANS, DIR_ALIGN, DIR_SPLIT):
    d.mkdir(parents=True, exist_ok=True)


# --------------------------- utils ---------------------------

def read_urls(txt_path: Path) -> List[str]:
    urls, seen = [], set()
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s not in seen:
            urls.append(s); seen.add(s)
    return urls

def base_and_id(stem: str):
    # stem format: <video_id>_<title>
    if "_" in stem:
        i = stem.find("_")
        return stem, stem[:i]
    return stem, stem  # fallback


# --------------------------- romanization (Arabic → Latin) ---------------------------

# Include Arabic + Supplement + Extended-A + Presentation Forms A/B
ARABIC_REGEX = re.compile(
    r'['
    r'\u0600-\u06FF'  # Arabic
    r'\u0750-\u077F'  # Arabic Supplement
    r'\u08A0-\u08FF'  # Arabic Extended-A
    r'\uFB50-\uFDFF'  # Presentation Forms-A
    r'\uFE70-\uFEFF'  # Presentation Forms-B
    r']'
)

def romanize_text(text: str) -> str:
    """
    Romanize Arabic portions; leaves Latin text as-is.
    Tries arabic2latin -> romanize3 -> unidecode.
    """
    if not ARABIC_REGEX.search(text or ""):
        return text
    try:
        from arabic2latin import arabic_to_latin
        out = arabic_to_latin(text)
        if isinstance(out, str) and out.strip():
            return out
    except Exception:
        pass
    try:
        from romanize3 import romanize
        out = romanize(text)
        if isinstance(out, str) and out.strip():
            return out
    except Exception:
        pass
    try:
        from unidecode import unidecode
        return unidecode(text)
    except Exception:
        return text


# --------------------------- yt-dlp (cookies optional) ---------------------------

def probe_info(url: str, cookiefile: Optional[Path]) -> Dict:
    ydl_opts = {"quiet": True, "noprogress": True, "skip_download": True}
    if cookiefile:
        ydl_opts["cookiefile"] = str(cookiefile)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info or "id" not in info:
        raise RuntimeError(f"Cannot extract info: {url}")
    return {"id": info["id"], "title": info.get("title", info["id"])}

def find_downloaded(video_id: str) -> Optional[Path]:
    m = list(DIR_DOWN.glob(f"{video_id}_*.wav"))
    return m[0] if m else None

def download_to_wav(url: str, cookiefile: Optional[Path]) -> Path:
    meta = probe_info(url, cookiefile=cookiefile)
    exist = find_downloaded(meta["id"])
    if exist:
        print(f"[download] exists -> {exist.name}")
        return exist

    outtmpl = str((DIR_DOWN / "%(id)s_%(title)s.%(ext)s").as_posix())
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bestaudio/best",
        "restrictfilenames": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "nopostoverwrites": False}
        ],
        "postprocessor_args": ["-ac", "1", "-ar", "24000"],  # mono 24k
        "quiet": True,
        "noprogress": True,
        "ignoreerrors": False,
        "concurrent_fragment_downloads": 3,
    }
    if cookiefile:
        ydl_opts["cookiefile"] = str(cookiefile)
        print(f"[cookies] using -> {cookiefile}")

    print(f"[download] {meta['id']} -> downloading")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    out = find_downloaded(meta["id"])
    if not out:
        raise FileNotFoundError(f"Downloaded WAV not found for {meta['id']}")
    return out


# --------------------------- faster-whisper ---------------------------

# GPU-only defaults per your request:
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
LANGUAGE = "id"

def load_fw(model_name: str) -> WhisperModel:
    """GPU-only (no CPU fallback)."""
    print(f"[fw] loading -> {model_name} (device={DEVICE}, compute_type={COMPUTE_TYPE})")
    return WhisperModel(model_name, device=DEVICE, compute_type=COMPUTE_TYPE)

def transcribe(model: WhisperModel, wav_path: Path) -> Dict:
    """
    Segments only (no word timestamps). We rely on Whisper's segmenting.
    """
    print(f"[asr] {wav_path.name}")
    seg_iter, info = model.transcribe(
        str(wav_path),
        language=LANGUAGE,           # fixed 'id'
        vad_filter=True,
        word_timestamps=False,       # segments only
        condition_on_previous_text=True,
    )
    segments = []
    for s in seg_iter:
        segments.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip()
        })
    return {"language": info.language, "duration": info.duration, "segments": segments}


# --------------------------- Alignment = pass-through (romanize Arabic only) ---------------------------

def align_passthrough(segs: List[Dict]) -> List[Dict]:
    """Keep start/end as-is; romanize any Arabic within text."""
    aligned = []
    for s in segs:
        txt = (s.get("text") or "").strip()
        if ARABIC_REGEX.search(txt):
            txt = romanize_text(txt)
        aligned.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": txt
        })
    return aligned


# --------------------------- split/export per video ---------------------------

def export_per_video(wav_path: Path, aligned: List[Dict]) -> int:
    """
    Export 1:1 with aligned segments (skip duration < 1.0s):
      output/splitted/[base]/[video_id]_[NNNN].wav
      output/splitted/[base]/[video_id]_transcripts.txt  (CSV header: filename,duration,transcript)
    """
    base = wav_path.stem                                 # <id>_<title>
    base_name, vid_id = base_and_id(base)
    out_dir = DIR_SPLIT / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # idempotent: skip if any chunk already exists
    if any(out_dir.glob(f"{vid_id}_*.wav")):
        print(f"[split] exists -> {base_name}")
        return 0

    audio = AudioSegment.from_file(wav_path)
    csv_path = out_dir / f"{vid_id}_transcripts.txt"

    count = 0
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "duration", "transcript"])
        n = 1
        for seg in aligned:
            st, en = float(seg["start"]), float(seg["end"])
            dur = max(0.0, en - st)
            if dur < 1.0:  # filter out very short segments
                continue

            txt = (seg.get("text") or "").strip()

            fname = f"{vid_id}_{n:04d}.wav"
            out_wav = out_dir / fname
            clip = audio[int(round(st*1000)):int(round(en*1000))].set_channels(1).set_frame_rate(24000)
            clip.export(out_wav, format="wav")
            writer.writerow([fname, f"{dur:.3f}", txt])

            n += 1
            count += 1

    print(f"[split] wrote -> {count} chunks for {base_name}")
    return count


# --------------------------- summary (from CSVs only) ---------------------------

def compute_summary(total_links: int) -> Dict:
    """
    Build summary purely from transcript CSVs so we don't decode audio.
    """
    # counts by pipeline stage (still useful)
    downloaded_files = len(list(DIR_DOWN.glob("*.wav")))
    transcribed_files = len(list(DIR_TRANS.glob("*.json")))
    aligned_files = len(list(DIR_ALIGN.glob("*.json")))

    # collect durations from all *_transcripts.txt
    csv_files = list(DIR_SPLIT.glob("**/*_transcripts.txt"))
    all_durations = []

    for path in csv_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                # expect ["filename","duration","transcript"]; tolerate minor variants
                dur_idx = 1
                if header:
                    # find "duration" column if present
                    try:
                        dur_idx = [h.strip().lower() for h in header].index("duration")
                    except ValueError:
                        dur_idx = 1  # fallback to 2nd column
                for row in reader:
                    if not row or len(row) <= dur_idx:
                        continue
                    try:
                        d = float(row[dur_idx])
                        if d >= 0:
                            all_durations.append(d)
                    except Exception:
                        continue
        except Exception:
            continue

    split_chunks = len(all_durations)
    total_sec = sum(all_durations)
    if all_durations:
        dmin = min(all_durations)
        dmax = max(all_durations)
        dmean = total_sec / split_chunks
        srt = sorted(all_durations)
        mid = split_chunks // 2
        dmed = srt[mid] if split_chunks % 2 == 1 else (srt[mid - 1] + srt[mid]) / 2.0
    else:
        dmin = dmax = dmean = dmed = 0.0

    summary = {
        "total_links": total_links,
        "downloaded_files": downloaded_files,
        "transcribed_files": transcribed_files,
        "aligned_files": aligned_files,
        "split_chunks": split_chunks,
        "chunks_duration_seconds": {
            "sum": round(total_sec, 3),
            "min": round(dmin, 3),
            "max": round(dmax, 3),
            "mean": round(dmean, 3),
            "median": round(dmed, 3),
        },
        "output_total_hours": round(total_sec / 3600.0, 3),
        "csv_files": len(csv_files),
    }

    (BASE_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="YT -> WAV -> faster-whisper -> passthrough align -> split per video -> summary-from-CSV")
    ap.add_argument("--urls_file", type=str, default="links.txt", required=True, help="txt with one YouTube URL per line")
    ap.add_argument("--cookies", type=str, default=None, help="Path to cookies.txt (Netscape format)")
    ap.add_argument("--model", type=str, default="medium", help="faster-whisper model (fallback: large-v3)")
    args = ap.parse_args()

    # cookies
    cookiefile = None
    if args.cookies:
        cookiefile = Path(args.cookies).resolve()
        if not cookiefile.exists():
            raise FileNotFoundError(f"cookies file not found: {cookiefile}")

    urls = read_urls(Path(args.urls_file))
    if not urls:
        print("No URLs found.")
        return

    # STAGE 1: DOWNLOAD
    print("\n========== STAGE 1: DOWNLOAD ==========")
    for u in urls:
        try:
            download_to_wav(u, cookiefile=cookiefile)
        except Exception as e:
            print(f"[download][error] {u} -> {e}")

    # STAGE 2: TRANSCRIBE (segments only)
    print("\n========== STAGE 2: TRANSCRIBE ==========")
    model = load_fw(args.model)

    for wav_path in sorted(DIR_DOWN.glob("*.wav")):
        base = wav_path.stem
        json_path = DIR_TRANS / f"{base}.json"
        if json_path.exists():
            print(f"[asr] exists -> {json_path.name}")
            continue
        try:
            result = transcribe(model, wav_path)
        except Exception as e:
            print(f"[asr][error] {wav_path.name} -> {e}")
            continue
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[asr] saved -> {json_path.name}")

    # STAGE 3: ALIGN (pass-through + romanize)
    print("\n========== STAGE 3: ALIGN ==========")
    for trans_json in sorted(DIR_TRANS.glob("*.json")):
        base = trans_json.stem
        out_json = DIR_ALIGN / f"{base}.json"
        if out_json.exists():
            print(f"[align] exists -> {out_json.name}")
            continue
        try:
            data = json.loads(trans_json.read_text(encoding="utf-8"))
            segs = data.get("segments", []) or []
            aligned = align_passthrough(segs)
            out_json.write_text(json.dumps({"segments": aligned}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[align] saved -> {out_json.name}")
        except Exception as e:
            print(f"[align][error] {base} -> {e}")

    # STAGE 4: SPLIT (export per video folder + transcripts CSV)
    print("\n========== STAGE 4: SPLIT ==========")
    for wav_path in sorted(DIR_DOWN.glob("*.wav")):
        base = wav_path.stem
        aligned_json = DIR_ALIGN / f"{base}.json"
        if not aligned_json.exists():
            print(f"[split] skip (no aligned JSON) -> {base}")
            continue

        try:
            aligned = json.loads(aligned_json.read_text(encoding="utf-8")).get("segments", []) or []
            export_per_video(wav_path, aligned)
        except Exception as e:
            print(f"[split][error] {base} -> {e}")

    # SUMMARY (from CSVs)
    print("\n========== SUMMARY ==========")
    summary = compute_summary(total_links=len(urls))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
