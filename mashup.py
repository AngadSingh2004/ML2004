"""Since youTube's structure or data response format ha changed so neither pytube nor yt-dl work so webapp cant be made"""

import sys
import os
from pytube import YouTube
from moviepy.editor import AudioFileClip
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

def download_videos(singer_name, num_videos):
    """Download N videos of the given singer from YouTube."""
    print(f"Searching and downloading {num_videos} videos of {singer_name}...")
    video_urls = [
        "https://www.youtube.com/watch?v=XXXX",
        "https://www.youtube.com/watch?v=YYYY",  
    ]
    downloaded_files = []
    for i, url in enumerate(video_urls[:num_videos]):
        try:
            yt = YouTube(url)
            stream = yt.streams.filter(only_audio=True).first()
            file_path = stream.download(filename=f"{singer_name}_{i}.mp4")
            downloaded_files.append(file_path)
            print(f"Downloaded: {file_path}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    
    return downloaded_files

def convert_videos_to_audio(video_files):
    """Convert downloaded video files to audio."""
    audio_files = []
    for video_file in video_files:
        audio_file = video_file.replace(".mp4", ".mp3")
        try:
            clip = AudioFileClip(video_file)
            clip.write_audiofile(audio_file)
            audio_files.append(audio_file)
            clip.close()
        except Exception as e:
            print(f"Error converting {video_file} to audio: {e}")
    
    return audio_files

def cut_audio_files(audio_files, duration):
    """Cut the first Y seconds from all audio files."""
    cut_audio_files = []
    for audio_file in audio_files:
        try:
            audio = AudioSegment.from_mp3(audio_file)
            cut_audio = audio[:duration * 1000] 
            cut_audio_file = audio_file.replace(".mp3", f"_cut_{duration}s.mp3")
            cut_audio.export(cut_audio_file, format="mp3")
            cut_audio_files.append(cut_audio_file)
        except CouldntDecodeError as e:
            print(f"Error processing {audio_file}: {e}")
    
    return cut_audio_files

def merge_audios(cut_audio_files, output_file):
    """Merge all the cut audio files into a single file."""
    combined_audio = AudioSegment.empty()
    for audio_file in cut_audio_files:
        audio = AudioSegment.from_mp3(audio_file)
        combined_audio += audio

    combined_audio.export(output_file, format="mp3")
    print(f"Audio files merged into {output_file}")

def main():
    if len(sys.argv) != 5:
        print("Usage: python 101556.py <SingerName> <NumberOfVideos> <AudioDuration> <OutputFileName>")
        sys.exit(1)
    
    singer_name = sys.argv[1]
    try:
        num_videos = int(sys.argv[2])
        audio_duration = int(sys.argv[3])
    except ValueError:
        print("Error: NumberOfVideos and AudioDuration must be integers.")
        sys.exit(1)
    output_file = sys.argv[4]
    
    if num_videos <= 10 or audio_duration <= 20:
        print("Error: NumberOfVideos must be greater than 10 and AudioDuration must be greater than 20 seconds.")
        sys.exit(1)
    video_files = download_videos(singer_name, num_videos)
    
    audio_files = convert_videos_to_audio(video_files)
    
    cut_audio_files_list = cut_audio_files(audio_files, audio_duration)
    
    merge_audios(cut_audio_files_list, output_file)

    print("Mashup completed successfully!")

if __name__ == "__main__":
    main()
