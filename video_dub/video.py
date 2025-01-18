from video_dub.audio import main as gen_audio
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import cv2
import math
import numpy as np
from argparse import ArgumentParser

LATENT_SYNC_PATH = Path("/workspace/LatentSync")


def get_frame_number(time, fps):
    return int(time * fps)


def read_n_next_frames(cap, n, get_scores=False):
    """
    Read n frames from a video capture object.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Video capture object
    n : int
        Number of frames to read
    get_scores : bool, optional
        If True, return frame scores

    Returns
    -------
    frames : List[np.ndarray]
        List of frames
    scores : List[float]
        List of frame scores, if `get_scores` is `True`
    """
    frames = []
    scores = [0]
    prev_frame = None
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if get_scores and prev_frame is not None:
            diff = cv2.absdiff(frame, prev_frame)
            score = np.mean(diff)
            scores.append(score)
        prev_frame = frame

    ret = frames
    if get_scores:
        ret = (frames, scores)

    return ret


def interpolate_frames(frame1, frame2, factor):
    """
    Create intermediate frames between two frames using linear interpolation.
    """
    return cv2.addWeighted(frame1, 1 - factor, frame2, factor, 0)


def gen_video_cv(input_path, output_path, gen_audio_spans, orig_audio_spans):
    """
    Generate a new video by editing the original video based on the audio spans.

    If the generated audio is shorter than the original audio, the video is squeezed.
    If the generated audio is longer than the original audio, the video is extended.

    Parameters
    ----------
    input_path : str
        Path to the input video file
    output_path : str
        Path to the output video file
    gen_audio_spans : List[Tuple[float, float]]
        List of audio spans for the generated audio
    orig_audio_spans : List[Tuple[float, float]]
        List of audio spans for the original audio
    target_audio_duration : float
        Target audio duration for the generated audio
    """

    #  Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / input_fps

    final_frames = []
    prev_span_end = 0
    start_frame = 0
    end_frame = 0
    # Iterate over edited spans
    # When we encounter a new span, we read frames from the previous
    # span end to the current span start. Then we read frames corresponding
    # to the edited span.
    # If edited span is shorter than original span, we reduce frames by
    # dropping frames with less significant changes.
    # If edited span is longer than original span, we extend frames by
    # frame interpolation.
    for orig_span, gen_span in zip(orig_audio_spans, gen_audio_spans):
        orig_span_start, orig_span_end = orig_span

        orig_span_len = orig_span_end - orig_span_start
        gen_span_len = gen_span[1] - gen_span[0]

        # Unedited section. We take as much as frames as we can.
        if orig_span_start > prev_span_end:
            start_frame = end_frame
            end_frame = get_frame_number(orig_span_start, input_fps)
            final_frames.extend(read_n_next_frames(cap, end_frame - start_frame))

        prev_span_end = orig_span_end
        # Delete scenario
        if gen_span_len <= 0.1:
            start_frame = end_frame
            end_frame = get_frame_number(orig_span_end, input_fps)
            read_n_next_frames(cap, end_frame - start_frame)
            continue

        # Edited section
        start_frame = end_frame
        if gen_span_len <= orig_span_len:
            # Reduce frames
            end_frame = get_frame_number(orig_span_end, input_fps)
            frames, frame_scores = read_n_next_frames(
                cap, end_frame - start_frame, get_scores=True
            )
            frame_scores = frame_scores / np.max(frame_scores)
            desired_frames = get_frame_number(gen_span_len, input_fps) + 1

            #  Select frames based on scores
            selected_frames = [frames[0]]  # Always include first frame
            total_score = np.sum(frame_scores)
            frames_per_score = (
                desired_frames - 2
            ) / total_score  # -2 to reserve first and last frames

            accumulated_frames = 0
            for i, score in enumerate(frame_scores[1:-1], 1):
                accumulated_frames += score * frames_per_score
                if accumulated_frames >= 1:
                    selected_frames.append(frames[i])
                    accumulated_frames -= 1

            selected_frames.append(frames[-1])  # Always include last frame
            num_selected = len(selected_frames)
            deficit = desired_frames - num_selected

            # If we have a deficit, we interpolate between the last two frames
            if deficit > 0:
                last_frame = selected_frames.pop(-1)
                deficit_frames = []
                for factor in range(deficit):
                    deficit_frames.append(
                        interpolate_frames(
                            selected_frames[-1], last_frame, factor / deficit
                        )
                    )
                selected_frames.extend(deficit_frames)
                selected_frames.append(last_frame)

            final_frames.extend(selected_frames)
        else:
            # Extend frames
            end_frame = get_frame_number(orig_span_end, input_fps)
            snippet_frames = read_n_next_frames(cap, end_frame - start_frame)
            num_original_frames = len(snippet_frames)

            desired_frames = get_frame_number(gen_span_len, input_fps)
            num_interp_frames = int(desired_frames / (num_original_frames - 1))
            extended_snippet_frames = []

            for i in range(num_original_frames - 1):
                frame1 = snippet_frames[i]
                frame2 = snippet_frames[i + 1]

                # Write the interpolated frames
                if i == num_original_frames - 2:
                    num_interp_frames = (
                        desired_frames - 1 - len(extended_snippet_frames)
                    )

                for j in range(num_interp_frames):
                    # Calculate interpolation factor
                    factor = j / num_interp_frames
                    interp_frame = interpolate_frames(frame1, frame2, factor)
                    extended_snippet_frames.append(interp_frame)

            extended_snippet_frames.append(snippet_frames[-1])
            final_frames.extend(extended_snippet_frames)

    # Leftover video
    if prev_span_end < video_duration:
        start_frame = end_frame
        end_frame = total_frames
        final_frames.extend(read_n_next_frames(cap, end_frame - start_frame))

    cap.release()

    # Write the final video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, input_fps, (frame_width, frame_height))
    for frame in final_frames:
        out.write(frame)
    out.release()


def gen_video_sync_lips(input_path, output_path, gen_audio_path, gen_audio_duration):
    """
    Use LatentSync to generate a new video with synced lips.

    When the generated audio is longer than the original audio, the video is
    extended by concatenating the original video with its reversed version.

    Parameters
    ----------
    input_path : str
        Path to the input video file
    output_path : str
        Path to the output video file
    gen_audio_path : str
        Path to the generated audio file
    gen_audio_duration : float
        Duration of the generated audio

    """
    # Get video duration
    cap = cv2.VideoCapture(input_path)
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / input_fps
    cap.release()

    if video_duration < gen_audio_duration:
        # Increase video duration.
        # Create sequence of original and reversed video clips to minimize
        # visual artifacts.

        reverse_video_path = output_path.parent / "input_reverse.mp4"
        temp_video_path = output_path.parent / "input_extended.mp4"
        revert_command = f"-i {input_path} -vf reverse -an {reverse_video_path}"
        mult_factor = math.ceil(gen_audio_duration / video_duration)
        concat_file = output_path.parent / "concat.txt"
        concat_content = [
            f"file '{input_path if i % 2 == 0 else reverse_video_path}'"
            for i in range(mult_factor)
        ]
        concat_file.write_text("\n".join(concat_content))
        concat_command = f"-f concat -safe 0 -i {concat_file} -c copy {temp_video_path}"
        commands = [revert_command, concat_command]
        print("\n".join(commands))
        for command in commands:
            subprocess.check_output(
                f"ffmpeg -y -hide_banner -loglevel error {command}", shell=True
            )
        input_path = temp_video_path

    command = f"""
        python -m scripts.inference \
        --unet_config_path "configs/unet/second_stage.yaml" \
        --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
        --inference_steps 20 \
        --guidance_scale 1.5 \
        --video_path "{input_path}" \
        --audio_path "{gen_audio_path}" \
        --video_out_path "{output_path}"
    """
    subprocess.check_call(command, shell=True, cwd=str(LATENT_SYNC_PATH))


def main(video_path, output_path, original_transcript, new_transcript, sync_lips=False):
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        input_audio_path = tmp_dir / "audio.wav"

        # Separate audio track from video
        subprocess.check_output(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                input_audio_path,
            ]
        )

        # Generate new audio
        gen_audio_path = output_path.parent / "gen_audio.wav"
        gen_audio_spans, orig_audio_spans, gen_audio_duration = gen_audio(
            input_audio_path, gen_audio_path, original_transcript, new_transcript
        )

        # Generate new video
        if sync_lips:
            gen_video_sync_lips(
                video_path, output_path, gen_audio_path, gen_audio_duration
            )
        else:
            gen_video_path = output_path.parent / "gen_video.mp4"
            gen_video_cv(
                video_path,
                gen_video_path,
                gen_audio_spans,
                orig_audio_spans,
            )

            # Combine new audio with video
            subprocess.check_output(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    gen_video_path,
                    "-i",
                    gen_audio_path,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-strict",
                    "experimental",
                    output_path,
                ]
            )


if __name__ == "__main__":
    argparser = ArgumentParser(help="Edit video based on new transcript")
    argparser.add_argument("--input_path", type=Path, required=True)
    argparser.add_argument("--output_path", type=Path, required=True)
    argparser.add_argument("--original_transcript", type=str, required=True)
    argparser.add_argument("--new_transcript", type=str, required=True)
    argparser.add_argument(
        "--sync_lips",
        action="store_true",
        help="When specified, the video will be lip-synced",
    )
    args = argparser.parse_args()

    main(
        args.input_path,
        args.output_path,
        args.original_transcript,
        args.new_transcript,
        args.sync_lips,
    )
