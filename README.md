# Video Dub

### Requirements
1. Docker
1. GPU with 25GB VRAM.

### Usage
```sh
docker pull eeishaan002/video_dub
docker run -it --rm --gpus=all -v `pwd`:/code eeishaan002/video_dub bash
cd /code
original_transcript="For a long time. Also this was the first time in disneyland for both of us. We've never been to disneyland in any other city. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a speed train that connects different cities in China."
new_single="For a long time. Also this was the first time in disneyland for both of us. We really like to travel and enjoy it alot. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a speed train that connects different cities in China."

# To generate video with traditional method of frame manipulation.
PYTHONPATH=. python video_dub/video.py \
    --input_path `pwd`/assets/demo5_video.mp4 \
    --output_path `pwd`/sample_out/demo5_out.mp4 \
    --original_transcript "$original_transcript" \
    --new_transcript "$new_single"

# To generate video with LatentSync model
PYTHONPATH=. python video_dub/video.py \
    --input_path `pwd`/assets/demo5_video.mp4 \
    --output_path `pwd`/sample_out/demo5_out_sync.mp4 \
    --original_transcript "$original_transcript" \
    --new_transcript "$new_single" \
    --sync_lips
```

### Generate all samples
```sh
docker pull eeishaan002/video_dub
docker run -it --rm --gpus=all -v `pwd`:/code eeishaan002/video_dub bash
bash -ex gen_samples.sh
```
Samples will be generated in `pwd`/sample_out directory.


### Traditional Method
1. We use SSR-Speech to run imputations on the original audio and generate new audio according to the new transcript.
1. Then we use Montreal Forced Aligner (MFA) to align the generated audio with the new transcript.
1. We also align the original audio to the original transcript in the same manner.
1. Using the two alignments, we map original video segments to their location in target video.


### LatentSync Method
1. We use SSR-Speech as in the traditional method above to generate desired audio.
1. Then we use LatentSync inference script to generate the lip-sync video.
1. For case where original video is shorter than the generated audio, we loop the video to extend it.
1. We create a concatenation of `[video, reversed_video, video, reversed_video ..]` until we reach the desired length.
