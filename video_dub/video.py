from video_dub.audio import main as gen_audio
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import cv2
import math


def get_frame_number(time, fps):
    return math.ceil(time * fps)


def read_n_next_frames(cap, n):
    frames = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def interpolate_frames(frame1, frame2, factor):
    """
    Create intermediate frames between two frames using linear interpolation.

    Args:
        frame1: First frame
        frame2: Second frame
        factor: Float between 0 and 1 indicating position between frames
    """
    return cv2.addWeighted(frame1, 1 - factor, frame2, factor, 0)


def gen_video_cv(
    input_path, output_path, gen_audio_spans, orig_audio_spans, target_audio_duration
):

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
    span_idx = 0
    start_frame = 0
    end_frame = 0
    for orig_span, gen_span in zip(orig_audio_spans, gen_audio_spans):
        orig_span_start, orig_span_end = orig_span

        orig_span_len = orig_span_end - orig_span_start
        gen_span_len = gen_span[1] - gen_span[0]

        if orig_span_start > prev_span_end:
            # Unedited section. We take as much as frames as we can.
            start_frame = end_frame
            end_frame = get_frame_number(orig_span_start, input_fps)
            final_frames.extend(read_n_next_frames(cap, end_frame - start_frame))
            span_idx += 1

        prev_span_end = orig_span_end
        if gen_span_len <= 0.1:
            continue

        # Edited section
        start_frame = end_frame
        if gen_span_len <= orig_span_len:
            # Reduce frames
            end_frame = start_frame + get_frame_number(gen_span_len, input_fps) - 1
            final_frames.extend(read_n_next_frames(cap, end_frame - start_frame))
            read_n_next_frames(
                cap, get_frame_number(orig_span_end, input_fps) - end_frame
            )
            end_frame = get_frame_number(orig_span_end, input_fps) - 1
        else:
            # Extend frames
            end_frame = get_frame_number(orig_span_end, input_fps) - 1
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

                    # Create and write interpolated frame
                    interp_frame = interpolate_frames(frame1, frame2, factor)
                    extended_snippet_frames.append(interp_frame)
            extended_snippet_frames.append(snippet_frames[-1])
            print("extended frames", len(extended_snippet_frames))
            print("desired frame", desired_frames)
            final_frames.extend(extended_snippet_frames)

        span_idx += 1

    if prev_span_end < video_duration:
        start_frame = end_frame  # get_frame_number(prev_span_end, input_fps)
        end_frame = total_frames
        final_frames.extend(read_n_next_frames(cap, end_frame - start_frame))
    cap.release()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, input_fps, (frame_width, frame_height))
    for frame in final_frames:
        out.write(frame)
    out.release()


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
        gen_video_cv(
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
    video_path = Path(__file__).parent.parent / "LatentSync/assets/demo3_video.mp4"
    output_path = Path(__file__).parent / "out.mp4"
    original_transcript = "For a long time. Also this was the first time in disneyland for both of us. We've never been to disneyland in any other city. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a speed train that connects different cities in China."
    new_transcript = "For a long time. Also this was the first time in disneyland for both of us. We really like to travel and enjoy it a lot. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a super fast boat that connects different cities in China."

    # original_transcript = "When I was a kid, I feel like you heard the thing, you heard the term, don't cry. you don't need to cry. crying is the most beautiful thing you can do. I encourage people to cry. I cry all the time. And I think it's the most healthy expression of how are you feeling and I, I sometimes wish."
    # new_transcript = "When I was a kid, I feel like you heard the thing, you heard the term, don't cry. It was said again and again, but crying is the most beautiful thing you can do. I encourage people to cry. I cry all the time. And I think it's the most healthy expression of how are you feeling and I, I sometimes wish."

    main(video_path, output_path, original_transcript, new_transcript)
# ==================ddddd==i==sss==============================
# edited durations [(5.53, 5.49), (5.97, 5.97), (6.43, 7.87)]
