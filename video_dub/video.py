from video_dub.audio import main as gen_audio
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import cv2


def get_video_duration(video_path):
    """
    Get video duration using OpenCV.
    Returns duration in seconds.
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("Could not open video file")

    # Get frame count and fps
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Calculate duration
    duration = frame_count / fps

    video.release()
    return duration


def seconds_to_ffmpeg_time(seconds: float) -> str:
    # round up to 3 decimal places
    seconds = round(seconds, 4)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60

    # Get milliseconds as integer
    milliseconds = int((seconds_remainder % 1) * 1000)
    seconds_int = int(seconds_remainder)

    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}.{milliseconds:03d}"


def gen_video(
    input_path, output_path, gen_audio_spans, orig_audio_spans, target_audio_duration
):
    final_span_files = []
    final_span_commands = []

    prev_span_end = 0
    temp_span_dir = Path(output_path).parent / "temp_spans"
    temp_span_dir.mkdir(exist_ok=True, parents=True)
    span_idx = 0

    video_duration = get_video_duration(input_path)

    for orig_span, gen_span in zip(orig_audio_spans, gen_audio_spans):
        orig_span_start, orig_span_end = orig_span

        orig_span_len = orig_span_end - orig_span_start
        gen_span_len = gen_span[1] - gen_span[0]

        if orig_span_start > prev_span_end:
            prev_span_file = temp_span_dir / f"span_{span_idx}.mp4"
            final_span_files.append(prev_span_file)
            final_span_commands.append(
                f"ffmpeg -y -hide_banner -loglevel error -i {input_path} -ss {seconds_to_ffmpeg_time(prev_span_end)} -to {seconds_to_ffmpeg_time(orig_span_start)} -an {prev_span_file}"
            )
            span_idx += 1

        prev_span_end = orig_span_end
        if gen_span_len <= 0.1:
            continue

        # Convert the original video to match the generated span len
        mult_factor = gen_span_len / orig_span_len
        gen_span_file = temp_span_dir / f"span_{span_idx}.mp4"
        final_span_files.append(gen_span_file)
        final_span_commands.append(
            f"ffmpeg -y -hide_banner -loglevel error -i {input_path} -ss {seconds_to_ffmpeg_time(orig_span_start)} -to {seconds_to_ffmpeg_time(orig_span_end)} -filter:v 'setpts={mult_factor}*PTS' -c:v libx264 -preset veryslow -crf 17 -an {gen_span_file}"
        )
        span_idx += 1

    if prev_span_end < video_duration:
        # mult_factor = (target_audio_duration - gen_span[1]) / (
        # video_duration - prev_span_end
        # )
        prev_span_file = temp_span_dir / f"span_{span_idx}.mp4"
        final_span_files.append(prev_span_file)
        final_span_commands.append(
            # f"ffmpeg -y -hide_banner -loglevel error -i {input_path} -ss {seconds_to_ffmpeg_time(prev_span_end)} -filter:v 'setpts={mult_factor}*PTS' -c:v libx264 -preset veryslow -crf 17 -an {prev_span_file}"
            f"ffmpeg -y -hide_banner -loglevel error -i {input_path} -ss {seconds_to_ffmpeg_time(prev_span_end)} -to {seconds_to_ffmpeg_time(video_duration)} -an {prev_span_file}"
        )

    print(final_span_commands)
    for command in final_span_commands:
        subprocess.check_output(command, shell=True)

    # Concatenate all the spans into the final video
    concat_file = temp_span_dir / "concat.txt"
    concat_file.write_text("\n".join([f"file '{f}'" for f in final_span_files]))
    concat_command = f"ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i {concat_file} -c copy {output_path}"
    subprocess.check_output(concat_command, shell=True)


def main(video_path, output_path, original_transcript, new_transcript):
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

        gen_audio_path = output_path.parent / "gen_audio.wav"

        # Generate new audio
        gen_audio_spans, orig_audio_spans, gen_audio_duration = gen_audio(
            input_audio_path, gen_audio_path, original_transcript, new_transcript
        )

        print("gen audio spans", gen_audio_spans)
        print("orig audio spans", orig_audio_spans)

        # Generate new video
        gen_video_path = output_path.parent / "gen_video.mp4"
        gen_video(
            video_path,
            gen_video_path,
            gen_audio_spans,
            orig_audio_spans,
            gen_audio_duration,
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
    video_path = Path(__file__).parent.parent / "LatentSync/assets/demo5_video.mp4"
    output_path = Path(__file__).parent / "out.mp4"
    # original_transcript = "For a long time. Also this was the first time in disneyland for both of us. We've never been to disneyland in any other city. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a speed train that connects different cities in China."
    # new_transcript = "For a long time. Also this was not our first time in disneyland. We been to disneyland in another city. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a speed train that connects a lot of different cities in China."

    original_transcript = "When I was a kid, I feel like you heard the thing, you heard the term, don't cry. you don't need to cry. crying is the most beautiful thing you can do. I encourage people to cry. I cry all the time. And I think it's the most healthy expression of how are you feeling and I, I sometimes wish."
    new_transcript = "When I was a kid, I feel like you heard the thing, you heard the term, don't cry. It was said again and again, but crying is the most beautiful thing you can do. I encourage people to cry. I cry all the time. And I think it's the most healthy expression of how are you feeling and I, I sometimes wish."

    main(video_path, output_path, original_transcript, new_transcript)
# ==================ddddd==i==sss==============================
# edited durations [(5.53, 5.49), (5.97, 5.97), (6.43, 7.87)]
