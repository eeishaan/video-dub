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

## Assumptions and Limitations
1. In cases where aligner fails to give us a tight alignment, the alignment for un-edited segments go out of sync.
2. SSR-Speech can only handle upto 3 edited segments. We also observed slight degradation when working with longer audio.

## Samples
<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td><b>Original Transcript</b></td>
        <td><b>New Transcript</b></td>
        <td><b>Videos</b></td>
  </tr>
  <tr>
    <td> For a long time. Also this was the first time in disneyland for both of us. We've never been to disneyland in any other city. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a speed train that connects different cities in China. </td>
    <td> For a long time. Also this was the first time in disneyland for both of us. <b>We really like to travel and enjoy it alot.</b> So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a speed train that connects different cities in China.</td>
    <td style="text-align: center;">
      <b>Original Video</b>
      <video src=https://github.com/user-attachments/assets/819df100-64ca-4236-b7ac-18ecec0919a0 controls preload></video>
      <b>Traditional Method</b>
      <video src=https://github.com/user-attachments/assets/7fd927fd-b8b5-484e-8f6f-c85f89139e42 controls preload></video>
      <b>Generative Method</b>
      <video src=https://github.com/user-attachments/assets/0e27f10d-40ef-4335-92e8-e818dcabe5fb controls preload></video>
    </td>
  </tr>
  <tr>
    <td> For a long time. Also this was the first time in disneyland for both of us. We've never been to disneyland in any other city. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a speed train that connects different cities in China. </td>
    <td> For a long time. Also this was <b>going to be a new day</b> in disneyland for both of us. We've never been to disneyland in any other city. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a <b>super fast train</b> that connects different cities in China.</td>
    <td>
      <b>Original Video</b>
       <video src=https://github.com/user-attachments/assets/819df100-64ca-4236-b7ac-18ecec0919a0 controls preload></video>
      <b>Traditional Method</b>
      <video src=https://github.com/user-attachments/assets/be962c4d-f312-4040-9c68-8d3cffa6aefd controls preload></video>
      <b>Generative Method</b>
      <video src=https://github.com/user-attachments/assets/a95f2c96-0d3f-4fa8-a0b0-22f10097afe1 controls preload></video>
    </td>
  <tr>
    <td> Still with me? Okay, now comes the fun part. Let's build you a wolf avatar. Youll be able to choose the fur colour, tail length, floppy or pointy ears. I choose floppy. And the colour of your liking. Great. Almost ready to go. Now that your wolf page is set up, it's time to invite your friends. So now click save. </td>
    <td> Still with me? Okay, now comes the fun part. Let's build you a wolf avatar. Youll be able to choose the <b>size of the head and shape of the nose.</b> I choose floppy. And the colour of your liking. Great. Almost ready to go. Now that your wolf page is set up, it's time to invite your friends. So now click save. </td>
    <td>
        <b>Original Video</b>
       <video height=100px src=https://github.com/user-attachments/assets/88df00ec-4965-43c9-aa28-7960b1c6d297 controls preload></video>
      <b>Traditional Method</b>
      <video src=https://github.com/user-attachments/assets/e9dfa0d1-dd54-42cb-923d-2e2a91db6b56 controls preload></video>
      <b>Generative Method</b>
      <video src=https://github.com/user-attachments/assets/62fc0efe-c533-4d42-bc92-a46ab37aacb0 controls preload></video>
    </td>
  </tr>
  <tr>
    <td> Still with me? Okay, now comes the fun part. Let's build you a wolf avatar. Youll be able to choose the fur colour, tail length, floppy or pointy ears. I choose floppy. And the colour of your liking. Great. Almost ready to go. Now that your wolf page is set up, it's time to invite your friends. So now click save. </td>
    <td> Still with me? Okay, now comes the fun part. Let's build you a wolf avatar. Youll be able to choose the <b>size of the head and shape of the nose.</b> I choose floppy. And the colour of your liking. Great. <b>Now you are going to wait a little bit.</b> Now that your wolf page is set up, it's time to invite your friends. So now click save. </td>
    <td>
        <b>Original Video</b>
       <video src=https://github.com/user-attachments/assets/88df00ec-4965-43c9-aa28-7960b1c6d297 controls preload></video>
      <b>Traditional Method</b>
      <video src=https://github.com/user-attachments/assets/668aa62c-d773-43b7-8120-dd1d8f825715 controls preload></video>
      <b>Generative Method</b>
      <video src=https://github.com/user-attachments/assets/26a738af-7399-4986-abc3-e851c8d49fd0 controls preload></video>
    </td>
  </tr>
  <tr>
    <td> Pretty silly. Little fucking fool. Now a word of caution. Do not ask people this if you are not fully ready to hear the answer. I was and am really down for criticism of myself as a human being because I want to be better. I am just naturally curious what people think of me. And I had the time to mentally prepare to hear their opinions without any defense </td>
    <td> Pretty silly. Little fucking fool. <b>Let me give you some advice first.</b> Do not ask people this if you are not fully ready to hear the answer. I was and am really down for criticism of myself as a human being because I want to be better. I am just naturally curious what people think of me. And I had the time to mentally prepare to hear their opinions without any defense </td>
    <td>
        <b>Original Video</b>
       <video src=https://github.com/user-attachments/assets/65a47e43-9f6c-4135-84f2-9dbd79c3a38a controls preload></video>
      <b>Traditional Method</b>
      <video src=https://github.com/user-attachments/assets/5db9cbd3-d3cc-43e5-81a4-128388ef62b0 controls preload></video>
      <b>Generative Method</b>
      <video src=https://github.com/user-attachments/assets/19bf7b5e-d3a8-4351-a83a-e441fe96cfcb controls preload></video>
    </td>
  </tr>
  <tr>
    <td> Pretty silly. Little fucking fool. Now a word of caution. Do not ask people this if you are not fully ready to hear the answer. I was and am really down for criticism of myself as a human being because I want to be better. I am just naturally curious what people think of me. And I had the time to mentally prepare to hear their opinions without any defense </td>
    <td> Pretty silly. Little fucking fool. <b>Let me give you some advice first.</b> Do not ask people this if you are not fully ready to hear the answer. I was and am really down for criticism of myself as a human being because I want to be better. I am just naturally curious what people <b>have to say about me.</b> And I had the time to mentally prepare to hear their opinions without any defense </td>
    <td>
        <b>Original Video</b>
       <video src=https://github.com/user-attachments/assets/65a47e43-9f6c-4135-84f2-9dbd79c3a38a controls preload></video>
      <b>Traditional Method</b>
       <video src=https://github.com/user-attachments/assets/2afb0b18-040f-451f-a426-e7391f8cf561 controls preload></video>
      <b>Generative Method</b>
       <video src=https://github.com/user-attachments/assets/27d506a7-1dad-43aa-a561-6090efac5fea controls preload></video>
    </td>
</table>
