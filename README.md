# Video Dub
Automatically edit talking-head style videos.

## Inputs
1. A talking-head style video with text transcript of its audio.
1. An edited transcript.

## Methods
### Traditional Method
1. We use [SSR-Speech](https://github.com/WangHelin1997/SSR-Speech) to generate audio for new text portions while using the unchanged portions as conditioning. This gives us new audio matching the new transcript.
1. Then we use Montreal Forced Aligner (MFA) to align the generated audio with the new transcript.
1. We also align the original audio to the original transcript in the same manner.
1. Using the two alignments, we map original video segments to their location in target video.
1. We use this information to keep the un-edited video segments aligned to the un-edited audio segments.
1. We use the following strategy.
    1. If an original video segment maps to a shorter segment in the target video, we drop frames from the original segment.
    1. If an original video segment maps to a longer segment in the target video, we use frame interpolation to extend the segment.

### LatentSync Method
1. In the traditional method above, the edited video segments can end up looking unaligned to the audio.
1. Thus we use a generative approach, [LatentSync](https://github.com/bytedance/LatentSync), to match narrator's mouth movements to the corresponding audio.
1. For cases where original video is shorter than the generated audio, we loop the video to extend it. This creates enough input video which can be modified to match the generated audio's length.
1. To ensure no abrupt changes occur during loops, loop is created by concatenating reversed video with the input video in an interleaving pattern like `[video, reversed_video, video, reversed_video ..]` until we reach the desired length.


## Usage

1. Pull pre-built docker image
    ```sh
    docker pull eeishaan002/video_dub
    ```

1. Mount the current repository and create a container with the pre-built image
    ```sh
    docker run -it --rm --gpus=all -v `pwd`:/code eeishaan002/video_dub bash
    ```

1. Prepare inputs
    ```sh
    cd /code
    original_transcript="For a long time. Also this was the first time in disneyland for both of us. We've never been to disneyland in any other city. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a speed train that connects different cities in China."
    new_single="For a long time. Also this was the first time in disneyland for both of us. We really like to travel and enjoy it alot. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a speed train that connects different cities in China."
    ```

1. Generate video with traditional frame manipulation.
    ```sh
    PYTHONPATH=. python video_dub/video.py \
        --input_path `pwd`/assets/demo5_video.mp4 \
        --output_path `pwd`/sample_out/demo5_out.mp4 \
        --original_transcript "$original_transcript" \
        --new_transcript "$new_single"
    ```

1. Generate video with LatentSync model.
    ```sh
    PYTHONPATH=. python video_dub/video.py \
        --input_path `pwd`/assets/demo5_video.mp4 \
        --output_path `pwd`/sample_out/demo5_out_sync.mp4 \
        --original_transcript "$original_transcript" \
        --new_transcript "$new_single" \
        --sync_lips
    ```

1. Generate all samples for demo. Samples will be generated in `sample_out` directory.
    ```sh
    bash -ex gen_samples.sh
    ```

1. You can also build the docker image again if you are unable to pull the image.
    ```sh
    docker build -t video_dub .
    ```

### Requirements
1. Docker
1. GPU with 25GB VRAM.
