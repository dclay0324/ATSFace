# NCCU Dataset

## Table of Contents
1. [Description](##Description)
2. [Datasets](##Datasets)
3. [Usage](##Usage)
4. [Citation](##Citation)
5. [Contact](##Contact)

## Description
This dataset is derived from an experiment involving university students. Initially, they answered questions about school life and finance. Later, they crafted fictitious narratives about chosen topics and then provided truthful narratives about another topic.

**Structure:**
1. Initial Questions: General questions about school life and finance.
2. Fictitious Narratives: Fabricated stories about topics including major, club experiences, internships, travel experiences, and personal hobbies.
3. Truthful Narratives: Honest stories about another chosen topic.

**Recording:**
Recordings were done using an iPhone 14 Pro in 1080p HD/30fps format. Participants responded to the moderator's questions in Chinese, exhibiting various facial expressions and behaviors.

**Dataset Composition:**
- Total Videos: 309 (147 deceptive, 162 truthful)
- Average Duration: 23.32 seconds (Range: 10.53 to 49.73 seconds)
- Participants: 23 unique male and 13 unique female speakers.

**Transcription:**
Transcripts of statements were made using the CapCut ASR system, retaining fillers and repeated words. The dataset contains:

- Total Words: 35,069 (1,403 unique words)
- Average Words/Transcript: 113


## Datasets
**Visual:**
- facenet_128: Extract visual features by detecting faces using RetinaFace, and transforming into 128-dimensional vectors using a pretrained FaceNet model.

**Audio:**
- mfcc: Sample the audio features and transform them into MFCC (Mel-frequency cepstral coefficients).
- mfcc_0.2: Sample the audio features and compute the average MFCCs for every 0.2 seconds in the video.

**Transcription:**
- srt_files: Convert the video to transcription by ASR system as SRT (SubRip Text). The description of SRT structure is below.
- chinese_bert_perword: Tokenize transcriptions using the Chinese BERT tokenizer, processing the text word-by-word. These tokens are passed through CKIP Lab's pretrained Chinese BERT model, producing a 768-dimensional vector for each word.
- chinese_bert_persentence: As above, but it's done sentence-by-sentence.

**SRT Structure**
- Sequence Number: A numeric counter identifying each sequential subtitle.
- Time Codes: Start and end times for when the subtitle should appear on screen, separated by an arrow (-->). The time is usually formatted as hours\:minutes\:seconds,milliseconds (e.g., 00:01:15,000 --> 00:01:20,000).
- Subtitle Text: The actual text of the subtitle. It can span multiple lines.
- Blank Line: A blank line indicating the end of this subtitle and the start of the next.


## Usage
This repository provides utility functions to process and extract features from videos and textual data. Here's how to utilize them:

**Reading Vectors from Text Files**

To read vectors from text files and get a list of vectors for each file:
```python
from functions import read_txt
file_paths = ['txt_path_1.txt', 'txt_path_2.txt']
vectors = read_txt(file_paths)
```

**Reading Transcriptions from SRT Files**

To extract transcriptions from subtitle files:
```python
from functions import read_srt
srt_paths = ['srt_path_1.srt', 'srt_path_2.srt']
transcriptions = read_srt(srt_paths)
```

**Extracting Face Embeddings from Videos using FaceNet**

To convert a list of video file paths into their corresponding face embeddings:
```python
from functions import video2facenet
video_paths = ['video_path_1.mp4', 'video_path_2.mp4']
face_embeddings = video2facenet(video_paths)
```

**Extracting MFCCs from Videos**

To extract Mel-frequency cepstral coefficients (MFCCs) from videos:
```python
from functions import video2mfccs
video_paths = ['video_path_1.mp4', 'video_path_2.mp4']
mfcc_vectors = video2mfccs(video_paths)
```

**Computing Average MFCCs for Specified Durations**

To compute average MFCCs for specific audio durations within videos:
```python
from functions import video2mfccs_mean
video_paths = ['video_path_1.mp4', 'video_path_2.mp4']
mfcc_mean_vectors = video2mfccs_mean(video_paths, period=0.2)
```

**Text Embeddings using BERT for Chinese Text**

To convert lists of Chinese textual data into embeddings:
```python
from functions import text2embed
textual_data = [['你好世界'], ['學習中文']]
embeddings = text2embed(textual_data)
```

Remember to ensure you have all the necessary dependencies installed, and adjust the provided paths in the examples to the actual paths of your data.

## Citation

## Contact
