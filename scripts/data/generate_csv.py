import pandas as pd
import sys

def create_csv(folder_name):

    segments = []

    with open(folder_name + '/segments.txt') as file_local:
        segments = file_local.readlines()

    segments = [seg.split(' ') for seg in segments]

    df_segment = pd.DataFrame(segments, columns = ['segment_id', 'file_id', 'start_time', 'end_time'])

    df_segment['end_time'] = df_segment['end_time'].str.strip()

    text = []
    with open(folder_name +'/text.txt') as file_local:
        text = file_local.readlines()

    text= [local_text.replace('\t',' ') for local_text in text]
    text = [local_text.split(' ',1) for local_text in text]

    df_text = pd.DataFrame(text, columns = ['segment_id', 'transcription'])
    df_text['transcription'] = df_text['transcription'].str.strip() 

    wavscp = []
    with open(folder_name+'/wav.scp.txt') as file:
        wavscp = file.readlines()
        
    wavscp = [utt.split('\t') for utt in wavscp]

    df_file_wav = pd.DataFrame(wavscp, columns =['file_id', 'file_path'])
    df_file_wav.file_path = df_file_wav.file_path.str.strip()

    df_merged = pd.merge(df_segment, df_text, on='segment_id' )
    df_merged = pd.merge(df_merged, df_file_wav, on='file_id' )

    df_merged['start_time'] = df_merged['start_time'].astype('float32')
    df_merged['end_time'] = df_merged['end_time'].astype('float32')

    df_merged['duration'] = df_merged.end_time - df_merged.start_time

    print("Total Duration is ", df_merged.duration.sum() / 3600)

    df_merged.to_csv(folder_name+'/'+folder_name+'.csv', index=False)

if __name__ == "__main__":
    folder_name = sys.argv[1]
    create_csv(folder_name)