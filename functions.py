import numpy as np
import cv2
import librosa
import librosa.display
import torch

from deepface import DeepFace
from transformers import (
  BertTokenizerFast,
  AutoModel
)

def read_txt(videos):
    '''
    This function reads vectors from given text files and 
    returns a list of these vectors for each file.
    
    Parameters:
        videos (list): A list of file paths to the text files containing the vectors.

    Returns:
        list of embeddings. A nested list where each sublist represents 
        a list of vectors extracted from each file.
    '''

    X = []
    for i in range(len(videos)):
        vectors = []

        f = open(videos[i])
        for line in f.readlines():
            vector = list(map(float, line.split(' ')[:-1]))
            vectors.append(vector)
        f.close()
        X.append(vectors)
    
    return X

def read_srt(srts):
    '''
    This function reads transcription from given srt files and 
    returns a list of these transcripts for each file.
    
    Parameters:
        srts (list): A list of file paths to the srt files.

    Returns:
        list of transcripts. A nested list where each sublist represents 
        a transcripts extracted from each file.
    '''
    
    X = []
    for i in range(len(srts)):
        vectors = []
        count = 0

        f = open(srts[i])
        for line in f.readlines():
            count += 1
            if count % 4 == 3:
                text = line.split('\n')[0]
                vectors.append(text)
        f.close()
        X.append(vectors)

    return X

def video2facenet(videos):
    '''
    Converts a list of video file paths to their corresponding face embeddings using the FaceNet model. 
    Each frame of a video is analyzed for faces, and the embeddings are generated using DeepFace.
    For frames where a face is not detected, a zero vector is appended.
    
    Parameters:
        videos (list): A list of file paths to the video files to be processed.
        
    Returns:
        list of embeddings: A nested list where each sublist contains face embeddings (vectors) 
        extracted from each video's frames. Each embedding is a 128-dimensional vector.
    '''

    vectors = []
    for i in range(len(videos)):
        print('video ' + str(i+1) + ' start!')
        total_frames, total_faces = 0, 0
        vector = []

        cap = cv2.VideoCapture(videos[i])
        while True:
            retval, img = cap.read()

            if not retval:
                break
            total_frames += 1
            
            try:
                face_obj = DeepFace.represent(img_path=img, model_name='Facenet', detector_backend="retinaface",
                                              enforce_detection=True, align=False)
                face = face_obj[0]['embedding']

                total_faces += 1
            except ValueError:
                if total_frames == 1:
                    face = np.zeros((128))

            vector.append(face)
        vectors.append(vector)
             
        print('video ' + str(i+1) + ': ' + str(total_frames) + ' frames, detect ' + str(total_faces) + ' faces')

    return vectors

def video2mfccs(videos, n_mfcc=20, hop_length=512):
    '''
    Converts a list of video file paths into their corresponding MFCCs (Mel-frequency cepstral coefficients).
    It loads the audio from the video and computes the MFCCs for the entire audio stream.
    
    Parameters:
        videos (list): A list of file paths to the video files to be processed.
        n_mfcc (int, optional): Number of MFCCs to compute. Default is 20.
        hop_length (int, optional): Number of samples between successive frames. Default is 512.
        
    Returns:
        list of embeddings: A list where each entry contains the MFCC vectors for the audio in a video. 
        Each MFCC vector is of length `n_mfcc`.
    '''
     
    vectors = []
     
    for i in range(len(videos)):
        y, sr = librosa.load(videos[i])
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length).T
            
        vectors.append(mfccs)
        
        print('video ' + str(i+1) + ': ' + str(len(mfccs)))
            
    return vectors

def video2mfccs_mean(videos, period, n_mfcc=20, hop_length=512):
    '''
    Converts a list of video file paths to their corresponding MFCCs (Mel-frequency cepstral coefficients) 
    for specified audio clip durations. It loads the audio from the video and computes the average MFCC 
    for each period in the audio.
    
    Parameters:
        videos (list): A list of file paths to the video files to be processed.
        period (float): Duration (in seconds) of each audio clip for which the MFCCs should be computed.
        n_mfcc (int, optional): Number of MFCCs to compute. Default is 20.
        hop_length (int, optional): Number of samples between successive frames. Default is 512.
        
    Returns:
        list of embeddings: A nested list where each sublist contains the average MFCC vectors 
        for each period of audio in a video. Each MFCC vector is of length `n_mfcc`.
    '''

    vectors = []
     
    for i in range(len(videos)):
        y, sr = librosa.load(videos[i])

        vector = []
        s = 0
        while int((s+period)*sr) <= len(y):
            mfccs = librosa.feature.mfcc(y=y[int(s*sr): int((s+period)*sr)], sr=sr, n_mfcc=n_mfcc, hop_length=hop_length).T
            mfccs = np.mean(mfccs, axis=0)
            
            vector.append(mfccs)
            s += period
        
        if int((s+period)*sr) > len(y):
            mfccs = librosa.feature.mfcc(y=y[int(s*sr):], sr=sr, n_mfcc=n_mfcc, hop_length=hop_length).T
            mfccs = np.mean(mfccs, axis=0)
            vector.append(mfccs)
            
        vectors.append(vector)
        
        print('video ' + str(i+1) + ': ' + str(len(vector)) + ' clips')
            
    return vectors

def text2embed(videos):
    '''
    Converts a list of lists of textual data into their corresponding embeddings per word
    using the BERT-based model for Chinese text.
    
    Parameters:
        videos (list of list of str): A nested list where each sublist contains 
        textual data to be converted into embeddings.
        
    Returns:
        list of embeddings: A nested list where each sublist contains the embeddings 
        for each word of textual data in a video.
    '''

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('ckiplab/bert-base-chinese')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vectors = []
    for i in range(len(videos)):
        vector = []
        for j in range(len(videos[i])):
            token = torch.Tensor([tokenizer(videos[i][j])['input_ids']]).int()
            embed = model(token.to(device)).last_hidden_state

            for emb in embed[0]:
                vector.append(emb.detach().numpy())
        vectors.append(vector)
    
    return vectors