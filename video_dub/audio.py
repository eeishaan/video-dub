from pathlib import Path
import torch
import subprocess
import textgrid
import librosa
import soundfile as sf
import torchaudio
from tempfile import TemporaryDirectory
from argparse import Namespace
import torchaudio
import sys

sys.path.insert(0, "/workspace/ssr")
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
from edit_utils_en import parse_edit_en, extract_words
from inference_scale import inference_one_sample
from models import ssr

MODEL_PATH = Path("/tmp/English.pth")
CODEC_PATH = Path("/tmp/wmencodec.th")


def textgrid_to_tuples(tg_path):
    """
    Convert a TextGrid file to a list of tuples
    Each tuple is of the form (start_time, end_time, word)

    Parameters
    ----------
    tg_path : str
        Path to the TextGrid file

    Returns
    -------
    list
        List of tuples
    """
    tg = textgrid.TextGrid.fromFile(tg_path)
    data = []
    for tier in tg.tiers:
        if tier.name.lower() != "words":  # Assuming the tier name is 'words'
            continue

        for interval in tier.intervals:
            if (
                not interval.mark.strip() or interval.mark == "<eps>"
            ):  # This skips empty intervals
                continue
            data.append((interval.minTime, interval.maxTime, interval.mark))
    return data


def get_word_alignment(audio_file, transcript):
    """
    Align an audio file with a transcript using MFA

    Parameters
    ----------
    audio_file : str
        Path to the audio file
    transcript : str
        Transcript of the audio

    Returns
    -------
    list
        List of tuples
    """
    with TemporaryDirectory() as tmpdir:
        transcript_file = Path(tmpdir) / "transcript.txt"
        transcript_file.write_text(transcript)

        try:
            # Run command and capture both stdout and stderr
            subprocess.run(
                [
                    "mfa",
                    "align_one",
                    str(audio_file),
                    str(transcript_file),
                    "english_mfa",
                    "english_mfa",
                    tmpdir,
                    "--beam",
                    "10",
                    "--clean",
                ],
                check=True,  # Raise CalledProcessError on non-zero exit
                text=True,  # Return strings instead of bytes
                capture_output=True,  # Capture both stdout and stderr
            )

        except subprocess.CalledProcessError as e:
            # Command failed - return the error output
            print(e.stderr, e.stdout)
            raise e

        except Exception as e:
            raise e

        tg_path = Path(tmpdir) / audio_file.with_suffix(".TextGrid").name
        data = textgrid_to_tuples(tg_path)
        return data


def load_models(model_path, codec_path, device):
    """
    Load the model and tokenizers

    Parameters
    ----------
    model_path : str
        Path to the model checkpoint
    codec_path : str
        Path to the codec file
    device : str
        Device to load the model on

    Returns
    -------
    tuple
        Tuple of the model, config, phn2num, audio_token
    """
    # Initialize models
    ckpt = torch.load(model_path, map_location="cpu")
    model = ssr.SSR_Speech(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    config = vars(model.args)
    phn2num = ckpt["phn2num"]
    model.to(device)
    model.eval()
    audio_tokenizer = AudioTokenizer(signature=codec_path)
    text_tokenizer = TextTokenizer(backend="espeak")
    return model, config, phn2num, audio_tokenizer, text_tokenizer


def resample_audio(audio, audio_out, target_sr=16_000):
    """
    Resample an audio file to `target_sr` and save it to `audio_out`

    Parameters
    ----------
    audio : str
        Path to the audio file
    audio_out : str
        Path to save the resampled audio
    target_sr : int, optional
        Target sample rate, by default 16_000
    """
    audio, _ = librosa.load(str(audio), sr=target_sr)
    sf.write(str(audio_out), audio, target_sr)


def get_mask_interval(data, word_span):
    """
    Get the start and end time of the mask interval.
    Copied from SSR-Speech/inference_v2.py

    Parameters
    ----------
    data : list
        List of tuples
    word_span : tuple
        Tuple of the form (start, end)

    Returns
    -------
    tuple
        Tuple of the form (start, end)
    """
    s, e = word_span[0], word_span[1]
    assert s <= e, f"s:{s}, e:{e}"
    assert s >= 0, f"s:{s}"
    assert e <= len(data), f"e:{e}"
    if e == 0:  # start
        start = 0.0
        end = float(data[0][0])
    elif s == len(data):  # end
        start = float(data[-1][1])
        end = float(data[-1][1])  # don't know the end yet
    elif s == e:  # insert
        start = float(data[s - 1][1])
        end = float(data[s][0])
    else:
        start = float(data[s - 1][1]) if s > 0 else float(data[s][0])
        end = float(data[e][0]) if e < len(data) else float(data[-1][1])

    return (start, end)


def get_target_span(input_spans, operations, num_orig_words):
    """
    Maps spans in the input transcript to spans in the target transcript.

    Parameters
    ----------
    input_spans : list
        List of tuples
    operations : list
        List of edit operations performed on the input transcript to synthesize
        the target transcript.
    num_orig_words : int
        Number of words in the original transcript

    Returns
    -------
    list
        List of tuples
    """

    target_spans = []

    span_offset = 0
    op_offset = 0

    for span in input_spans:
        start_span, end_span = span
        target_start = start_span + span_offset

        if end_span + span_offset >= num_orig_words:
            target_end = len(operations) - 1
            target_spans.append((target_start, target_end))
            continue

        source_span_end = end_span + span_offset + 1
        while source_span_end < len(operations) and (
            operations[source_span_end] == "i" or operations[source_span_end] == "d"
        ):
            source_span_end += 1

        ops = operations[start_span + op_offset : source_span_end]
        new_op_offset = sum([1 for op in ops if op == "i"])
        op_offset += new_op_offset

        ops = operations[start_span + op_offset : end_span + op_offset]
        span_offset += new_op_offset - sum([1 for op in ops if op == "d"])
        target_end = end_span + span_offset
        target_end = max(target_start, target_end)
        target_spans.append((target_start, target_end))
    return target_spans


def find_edits(orig_transcript, new_transcript, orig_audio):
    """
    Find edited portions in the original transcript.

    Code borrowed from SSR-Speech/inference_v2.py

    Parameters
    ----------
    orig_transcript : str
        Original transcript text
    new_transcript : str
        New transcript text
    orig_audio : str
        Path to the original audio file

    Returns
    -------
    mask_interval : torch.LongTensor
        Mask interval in codec space for the edited portion
    combined_target_spans : list
        List of tuples containing the start and end indices of the
        edited portion in the new transcript.
    combined_orig_spans : list
        List of tuples containing the start and end times of the
        edited portion in the original audio.
    """

    word_data = get_word_alignment(orig_audio, orig_transcript)

    num_orig_words = len(extract_words(orig_transcript))
    assert len(word_data) == num_orig_words, f"{len(word_data)} != {num_orig_words}"
    operations, orig_spans = parse_edit_en(orig_transcript, new_transcript)

    if len(orig_spans) > 3:
        raise RuntimeError("Current model only supports maximum 3 editings")

    starting_intervals = []
    ending_intervals = []
    for orig_span in orig_spans:
        start, end = get_mask_interval(word_data, orig_span)
        starting_intervals.append(start)
        ending_intervals.append(end)

    info = torchaudio.info(orig_audio)
    audio_dur = info.num_frames / info.sample_rate

    def combine_spans(spans, threshold=0.2):
        spans.sort(key=lambda x: x[0])
        combined_spans = []
        current_span = spans[0]

        merge_trail = []
        curr_trail = [0]
        for i in range(1, len(spans)):
            next_span = spans[i]
            if current_span[1] >= next_span[0] - threshold:
                current_span[1] = max(current_span[1], next_span[1])
                curr_trail.append(i)
            else:
                combined_spans.append(current_span)
                current_span = next_span
                merge_trail.append(curr_trail)
                curr_trail = [i]
        merge_trail.append(curr_trail)
        combined_spans.append(current_span)
        return combined_spans, merge_trail

    sub_amount = 0.12
    codec_sr = 50
    morphed_span = [
        [max(start - sub_amount, 0), min(end + sub_amount, audio_dur)]
        for start, end in zip(starting_intervals, ending_intervals)
    ]  # in seconds
    morphed_span, merged_span_trail = combine_spans(morphed_span, threshold=0.2)

    mask_interval = [
        [round(span[0] * codec_sr), round(span[1] * codec_sr)] for span in morphed_span
    ]
    mask_interval = torch.LongTensor(mask_interval)  # [M,2], M==1 for now

    # Map the edited spans in the original transcript to spans in new transcript
    target_spans = get_target_span(orig_spans, operations, num_orig_words)

    # Find original span timings + merge target spans
    combined_target_spans = []
    combined_orig_spans = []
    for merged_span in merged_span_trail:
        combined_target_spans.append(
            (target_spans[merged_span[0]][0], target_spans[merged_span[-1]][1])
        )
        combined_orig_spans.append(
            (
                word_data[orig_spans[merged_span[0]][0]][0],
                word_data[orig_spans[merged_span[-1]][1]][0],
            )
        )
    return mask_interval, combined_target_spans, combined_orig_spans


def sample(
    model,
    config,
    phn2num,
    text_tokenizer,
    audio_tokenizer,
    orig_audio,
    orig_transcript,
    new_transcript,
    mask_interval,
    device,
):
    """
    Generate new audio
    """
    decode_config = {
        "top_k": 0,
        "top_p": 0.8,
        "temperature": 1,
        "stop_repetition": 2,
        "kvcache": 1,
        "codec_audio_sr": 16_000,
        "codec_sr": 50,
    }
    new_audio = inference_one_sample(
        model,
        Namespace(**config),
        phn2num,
        text_tokenizer,
        audio_tokenizer,
        orig_audio,
        orig_transcript,
        new_transcript,
        mask_interval,
        cfg_coef=1.5,
        cfg_stride=5,
        aug_text=True,
        aug_context=False,
        use_watermark=False,
        tts=False,
        device=device,
        decode_config=decode_config,
    )
    # save segments for comparison
    new_audio = new_audio[0].cpu()
    return new_audio


def main(input_audio_path, output_audio_path, orig_transcript, new_transcript):

    model_path = MODEL_PATH
    codec_path = CODEC_PATH

    assert torch.cuda.is_available(), "CUDA is required for this script"
    device = "cuda"

    model, config, phn2num, audio_tokenizer, text_tokenizer = load_models(
        model_path, codec_path, device
    )

    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        resampled_audio_path = temp_dir / "resampled_audio.wav"
        resample_audio(input_audio_path, resampled_audio_path)

        orig_transcript = orig_transcript.lower().strip()
        new_transcript = new_transcript.lower().strip()

        mask_interval, edited_spans, original_spans = find_edits(
            orig_transcript, new_transcript, resampled_audio_path
        )

        new_audio = sample(
            model,
            config,
            phn2num,
            text_tokenizer,
            audio_tokenizer,
            resampled_audio_path,
            orig_transcript,
            new_transcript,
            mask_interval,
            device,
        )
    # Write output audio
    sf.write(output_audio_path.as_posix(), new_audio.squeeze().numpy(), 16_000)

    # Align new audio with new transcript. This is needed
    # to get the start and end times of the edited portions. This is useful
    # for video editing.
    new_word_alignment = get_word_alignment(output_audio_path, new_transcript)
    edited_durations = []
    edited_segments = []
    for span in edited_spans:
        start, end = span
        edited_durations.append(
            (new_word_alignment[start][0], new_word_alignment[end][0])
        )
        edited_segments.append(
            (new_word_alignment[start][2], new_word_alignment[end - 1][2])
        )

    # Free up memory
    del model, audio_tokenizer, text_tokenizer
    torch.cuda.empty_cache()

    return edited_durations, original_spans, new_audio.shape[-1] / 16_000


if __name__ == "__main__":
    audio_path = Path(__file__).parent / "ssr/demo/84_121550_000074_000000.wav"
    orig_transcript = "but when i had approached so near to them, the common object, which the sense deceives, lost not by distance any of its marks."
    new_transcript = "but when i saw the mirage of the lake in the distance, which the sense deceives, lost not by distance any marks or any project,"
    new_transcript = (
        "but when i know which the sense deceives, lost not by distance any its marks"
    )
    out_path = Path(__file__).parent / "new_audio.wav"

    target_span, input_span, duration = main(
        audio_path, out_path, orig_transcript, new_transcript
    )
    print("target_span", target_span)
    print("input_span", input_span)
    print("duration", duration)
