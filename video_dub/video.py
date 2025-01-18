from video_dub.audio import main as gen_audio
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import cv2
import math
import numpy as np


def get_frame_number(time, fps):
    # return math.ceil(time * fps)
    # return round(time * fps)
    return int(time * fps)


def read_n_next_frames(cap, n, get_scores=False):
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

        prev_span_end = orig_span_end
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
            print("red frames", len(frames))
            print("frame scores", len(frame_scores))
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
            print("desired squeezed frames", desired_frames)
            print("framed selected", len(selected_frames))
            final_frames.extend(selected_frames)
        else:
            # Extend frames
            end_frame = get_frame_number(orig_span_end, input_fps)
            snippet_frames = read_n_next_frames(cap, end_frame - start_frame)
            num_original_frames = len(snippet_frames)
            print("desired frames", end_frame - start_frame)
            print("read frames", num_original_frames)
            print("total frames", len(final_frames), "max frames", total_frames)
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
    video_path = Path(__file__).parent.parent / "LatentSync/assets/demo3_video.mp4"
    output_path = Path(__file__).parent / "out.mp4"
    original_transcript = "For a long time. Also this was the first time in disneyland for both of us. We've never been to disneyland in any other city. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a speed train that connects different cities in China."
    new_transcript = "For a long time. Also this was the first time in disneyland for both of us. We've never been to disneyland in any other city. So, first of all, because we live in another city, we need to register ourselves at the passport office. It's a speed train that connects different cities in China."
    # new_transcript = "For a long time. Also this was going to be a new day in disneyland for both of us. We've never been to disneyland in any other city. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a super fast train that connects different cities in China."

    video_path = Path(__file__).parent.parent / "LatentSync/assets/demo5_video.mp4"
    original_transcript = "When I was a kid, I feel like you heard the thing, you heard the term, don't cry. you don't need to cry. crying is the most beautiful thing you can do. I encourage people to cry. I cry all the time. And I think it's the most healthy expression of how are you feeling and I, I sometimes wish."
    new_transcript = "When I was a kid, I feel like you heard the thing, you heard the term, don't cry. It always kind of unsettling to me. crying is the most beautiful thing you can do. I encourage people to cry. It'a great habit after all. And I think it's the most healthy expression of how are you feeling and I, I sometimes wish."

    video_path = (
        Path(__file__).parent.parent / "LatentSync/assets/akana_shorter_norm.mp4"
    )
    original_transcript = "Pretty silly. Little fucking fool. Now a word of caution. Do not ask people this if you are not ready to hear the answer. I was and am really down for criticism of myself as a human being because I want to be better. I am just naturally curious what people think of me. And I had the time to mentally prepare to hear their opinions without any defense"
    new_transcript = "Pretty silly. Little fucking fool. Let me give you some advice first. Do not ask people this if you are not ready to hear the answer. I was and am really down for criticism of myself as a human being because I want to be better. I am just naturally curious what people have to say about me. And I had the time to mentally prepare to hear their opinions without any defense"

    video_path = Path(__file__).parent.parent / "LatentSync/assets/wolf_norm.mp4"
    original_transcript = "Still with me? Okay, now comes the fun part. Let's build you a wolf avatar. Youll be able to choose the fur colour, tail length, floppy or pointy ears. I choose floppy. And the colour of your liking. Great. Almost ready to go. Now that your wolf page is set up, it's time to invite your friends. So now click save."
    new_transcript = "Still with me? Okay, now comes the fun part. Let's build you a wolf avatar. Youll be able to choose the size of the head and shape of the nose. I choose floppy. And the colour of your liking. Great. Now you are going to wait a little bit. Now that your wolf page is set up, it's time to invite your friends. So now click save."
    main(video_path, output_path, original_transcript, new_transcript)
# ==================ddddd==i==sss==============================
# edited durations [(5.53, 5.49), (5.97, 5.97), (6.43, 7.87)]
