export PYTHONPATH=`pwd`
original_transcript="For a long time. Also this was the first time in disneyland for both of us. We've never been to disneyland in any other city. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a speed train that connects different cities in China."
new_single="For a long time. Also this was the first time in disneyland for both of us. We really like to travel and enjoy it alot. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a speed train that connects different cities in China."
new_double="For a long time. Also this was going to be a new day in disneyland for both of us. We've never been to disneyland in any other city. So, first of all, because we live in another city, we need to travel to Shanghai on Gaojia. It's a super fast train that connects different cities in China."
python video_dub/video.py \
    --input_path `pwd`/LatentSync/assets/demo3_video.mp4 \
    --output_path `pwd`/sample_out/demo5_single_sync.mp4 \
    --original_transcript "$original_transcript" \
    --new_transcript "$new_single" \
    --sync_lips
python video_dub/video.py \
    --input_path `pwd`/LatentSync/assets/demo3_video.mp4 \
    --output_path `pwd`/sample_out/demo5_double_sync.mp4 \
    --original_transcript "$original_transcript" \
    --new_transcript "$new_double" \
    --sync_lips
python video_dub/video.py \
    --input_path `pwd`/LatentSync/assets/demo3_video.mp4 \
    --output_path `pwd`/sample_out/demo5_single.mp4 \
    --original_transcript "$original_transcript" \
    --new_transcript "$new_single"
python video_dub/video.py \
    --input_path `pwd`/LatentSync/assets/demo3_video.mp4 \
    --output_path `pwd`/sample_out/demo5_double.mp4 \
    --original_transcript "$original_transcript" \
    --new_transcript "$new_double"

original_transcript="Pretty silly. Little fucking fool. Now a word of caution. Do not ask people this if you are not fully ready to hear the answer. I was and am really down for criticism of myself as a human being because I want to be better. I am just naturally curious what people think of me. And I had the time to mentally prepare to hear their opinions without any defense"
new_single="Pretty silly. Little fucking fool. Let me give you some advice first. Do not ask people this if you are not fully ready to hear the answer. I was and am really down for criticism of myself as a human being because I want to be better. I am just naturally curious what people think of me. And I had the time to mentally prepare to hear their opinions without any defense"
new_double="Pretty silly. Little fucking fool. Let me give you some advice first. Do not ask people this if you are not fully ready to hear the answer. I was and am really down for criticism of myself as a human being because I want to be better. I am just naturally curious what people have to say about me. And I had the time to mentally prepare to hear their opinions without any defense"
python video_dub/video.py \
    --input_path `pwd`/LatentSync/assets/akana_shorter_norm.mp4 \
    --output_path `pwd`/sample_out/akana_single_sync.mp4 \
    --original_transcript "$original_transcript" \
    --new_transcript "$new_single" \
    --sync_lips
python video_dub/video.py \
    --input_path `pwd`/LatentSync/assets/akana_shorter_norm.mp4 \
    --output_path `pwd`/sample_out/akana_double_sync.mp4 \
    --original_transcript "$original_transcript" \
    --new_transcript "$new_double" \
    --sync_lips
python video_dub/video.py \
    --input_path `pwd`/LatentSync/assets/akana_shorter_norm.mp4 \
    --output_path `pwd`/sample_out/akana_single.mp4 \
    --original_transcript "$original_transcript" \
    --new_transcript "$new_single"
python video_dub/video.py \
    --input_path `pwd`/LatentSync/assets/akana_shorter_norm.mp4 \
    --output_path `pwd`/sample_out/akana_double.mp4 \
    --original_transcript "$original_transcript" \
    --new_transcript "$new_double"

original_transcript="Still with me? Okay, now comes the fun part. Let's build you a wolf avatar. Youll be able to choose the fur colour, tail length, floppy or pointy ears. I choose floppy. And the colour of your liking. Great. Almost ready to go. Now that your wolf page is set up, it's time to invite your friends. So now click save."
new_single="Still with me? Okay, now comes the fun part. Let's build you a wolf avatar. Youll be able to choose the size of the head and shape of the nose. I choose floppy. And the colour of your liking. Great. Almost ready to go. Now that your wolf page is set up, it's time to invite your friends. So now click save."
new_double="Still with me? Okay, now comes the fun part. Let's build you a wolf avatar. Youll be able to choose the size of the head and shape of the nose. I choose floppy. And the colour of your liking. Great. Now you are going to wait a little bit. Now that your wolf page is set up, it's time to invite your friends. So now click save."
python video_dub/video.py \
    --input_path `pwd`/LatentSync/assets/wolf_norm.mp4 \
    --output_path `pwd`/sample_out/wolf_single_sync.mp4 \
    --original_transcript "$original_transcript" \
    --new_transcript "$new_single" \
    --sync_lips
python video_dub/video.py \
    --input_path `pwd`/LatentSync/assets/wolf_norm.mp4 \
    --output_path `pwd`/sample_out/wolf_double_sync.mp4 \
    --original_transcript "$original_transcript" \
    --new_transcript "$new_double" \
    --sync_lips
python video_dub/video.py \
    --input_path `pwd`/LatentSync/assets/wolf_norm.mp4 \
    --output_path `pwd`/sample_out/wolf_single.mp4 \
    --original_transcript "$original_transcript" \
    --new_transcript "$new_single"
python video_dub/video.py \
    --input_path `pwd`/LatentSync/assets/wolf_norm.mp4 \
    --output_path `pwd`/sample_out/wolf_double.mp4 \
    --original_transcript "$original_transcript" \
    --new_transcript "$new_double"
