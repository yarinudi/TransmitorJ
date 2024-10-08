import preprocessing
import numpy as np
import os
from models_new import GaitDetectorSSL,GaitChoreaDetectorSSL
import torch
from sklearn import metrics
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt
import ipdb
import wandb
import argparse

parser = argparse.ArgumentParser(description='hd_gait_detection_with_SSL')

parser.add_argument(
    '--cohort',
    type=str,
    default='hd',
    help='hd-huntington patients only/hc-healthy control only/mixed')

parser.add_argument(
    '--create-multi-class',
    action='store_true',
    default=True,
    help='if true multiclass of gait and chorea classification is applied')

parser.add_argument(
    '--preprocess-mode',
    action='store_true',
    default=False,
    help='if true the raw data is preprocessed')

parser.add_argument(
    '--cross-val-mode',
    action='store_true',
    default=False,
    help='if true training and validation of the model is applied')

parser.add_argument(
    '--eval-mode',
    action='store_true',
    default=False,
    help='if true evaluation of training results is applied')

parser.add_argument(
    '--gait-all-mode',
    action='store_true',
    default=False,
    help='if true include gait/non-gait window without chorea labels')

parser.add_argument(
    '--run-suffix',
    type=str,
    default='5sec_all',
    help='specify the run name')

parser.add_argument(
    '--wandb-flag',
    action='store_true',
    default=False,
    help='if true log to wandb')

parser.add_argument(
    '--model-type',
    type=str,
    default='segmentation',
    help='specify if segmentation or classification')

parser.add_argument(
    '--padding-type',
    type=str,
    default='triple_wind',
    help='specify if without_edges or triple_wind')


args = parser.parse_args()




VISUALIZE_ACC_VS_PRED_WIN = False
RAW_DATA_AND_LABELS_DIR = '/home/dafnas1/datasets/hd_dataset/lab_geneactive/synced_labeled_data_walking_non_walking'
#RAW_DATA_AND_LABELS_DIR = '/mlwell-data2/dafna/daily_living_data_array/HC'
#RAW_DATA_AND_LABELS_DIR = '/mlwell-data2/dafna/PD_data_and_labels/Data'
#PD_RAW_LABELS_DIR = '/mlwell-data2/dafna/PD_data_and_labels/labels'
#RAW_DATA_AND_LABELS_DIR = '/mlwell-data2/dafna/daily_living_data_array/PACE'
PROCESSED_DATA_DIR ='/mlwell-data2/dafna/daily_living_data_array/data_ready'
OUTPUT_DIR = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs'
VIZUALIZE_DIR = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/results_visualization/multiclass_hd_only/multiclass_separated_labels'
if args.cohort == 'pd_owly':
    SRC_SAMPLE_RATE = int(25) #hz
else:
    SRC_SAMPLE_RATE = int(100) #hz
STD_THRESH = 0.1
WINDOW_SIZE = int(30*10)
if args.model_type=='classification':
    WINDOW_OVERLAP=0
elif args.padding_type=='triple_wind':
    WINDOW_OVERLAP = int(30*5)
elif args.padding_type=='without_edges':
    WINDOW_OVERLAP = int(30*4)
else:
    WINDOW_OVERLAP=0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.wandb_flag:
        wandb.init(project='hd_gait_detection_with_ssl')
    if args.preprocess_mode:
        #iterate over subjects and preprocess the data
        win_data_all_sub = np.empty((0,3,WINDOW_SIZE)) 
        if args.model_type == 'classification':
            win_labels_all_sub = win_subjects = all_subjects = win_chorea_all_sub = win_shift_all_sub = np.empty((0,1))
        elif  args.model_type == 'segmentation':
             win_labels_all_sub = win_subjects = all_subjects = win_chorea_all_sub = win_shift_all_sub = np.empty((0,WINDOW_SIZE))
        StdIndex_all = inclusion_idx = original_data_len = np.empty((0,))
        win_video_time_all_sub = np.empty((0,1))
        NumWin = []
        for file in os.listdir(RAW_DATA_AND_LABELS_DIR):
            try:
                if args.cohort == 'hc':
                    if 'TCCO' in file or 'CF' in file:
                        data_file = np.load(os.path.join(RAW_DATA_AND_LABELS_DIR, file))
                    else:
                        continue
                if args.cohort == 'hd':
                    if 'TCCO' in file:
                        continue 
                    if 'WS' in file:
                        continue   
                    else:
                        data_file = np.load(os.path.join(RAW_DATA_AND_LABELS_DIR, file))
                if args.cohort == 'pd_owly':
                    acc_data = np.load(os.path.join(RAW_DATA_AND_LABELS_DIR, file),allow_pickle=True)
                    label_data = np.load(os.path.join(PD_RAW_LABELS_DIR, file.replace('data', 'labels')),allow_pickle=True)
                    data_file = {'arr_0': acc_data, 'arr_1': label_data}
                    
            except:
                print(f"cant open the file {file}")
                
                continue
            try:
                acc_data = data_file['arr_0'].astype('float')
            except:
                try:
                    # remove lines with empty string
                    acc_data = data_file['arr_0']
                    def is_numeric(s):
                        try:
                            float(s)
                            return True
                        except ValueError:
                            return False
                    # Convert non-numeric strings to np.nan
                    numeric_mask = np.array([[is_numeric(cell) for cell in row] for row in acc_data])
                    # numeric_mask = np.char.isnumeric(acc_data) | np.char.isdecimal(acc_data)
                    acc_data[~numeric_mask] = np.nan
                    acc_data = acc_data.astype('float')
                    acc_data = acc_data[~np.isnan(acc_data).any(axis=1)]
                    if len(acc_data) == 0:
                        continue
                except:  
                    print(f"failed to open {file}")                  
                    continue
            if args.cohort == 'pd_owly':
                labels = np.load(os.path.join(RAW_DATA_AND_LABELS_DIR, file),allow_pickle=True)
            labels = data_file.get('arr_1', None)
            chorea = data_file.get('arr_2', None)
            if args.cohort == 'hc' and chorea is not None:
               chorea[chorea==-1] = 0 
            video_time = data_file.get('arr_3', None)
            subject_name = file.split('.')[0]
            
            if args.cohort == 'pd_owly':
                acc_data = preprocessing.highpass_filter(data=acc_data,high_cut=0.2,sampling_rate=SRC_SAMPLE_RATE,order=4)
            else:
                acc_data = preprocessing.bandpass_filter(data=acc_data,low_cut=0.2,high_cut=15,sampling_rate=SRC_SAMPLE_RATE,order=4)
            ## apply resampling 

            acc_data,labels, chorea, video_time = preprocessing.resample(data=acc_data,labels=labels,chorea=chorea, video_time=video_time ,original_fs=SRC_SAMPLE_RATE,target_fs=30)

            
            ## deivide data and labels to fixed windows
            data, labels, chorea, video_time, shift, NumWinSub = preprocessing.data_windowing(data=acc_data, labels=labels, chorea=chorea, video_time=video_time, window_size = WINDOW_SIZE, window_overlap=WINDOW_OVERLAP,
                                                                                std_th=STD_THRESH,model_type=args.model_type)
            # Concat the data and labels of the different subjects
            
            win_data_all_sub = np.append(win_data_all_sub, data, axis=0)
            win_labels_all_sub = np.append(win_labels_all_sub, labels, axis=0)
            win_chorea_all_sub = np.append(win_chorea_all_sub, chorea, axis=0)
            win_shift_all_sub = np.append(win_shift_all_sub, shift, axis=0)
            win_video_time_all_sub = np.append(win_video_time_all_sub, video_time, axis=0)
            print(file,win_data_all_sub.shape)
            # Create subject vector that will use for group the data in the training
            subject = np.tile(subject_name, (len(labels), 1)).reshape(-1, 1)
            win_subjects = np.append(win_subjects, subject)

            # StdIndex_all = np.append(StdIndex_all, StdIndex, axis=0)
            # inclusion_idx = np.append(inclusion_idx.squeeze(), inclusion, axis=0)
            # original_data_len = np.append(original_data_len, len(StdIndex))
            NumWin.append(NumWinSub)

        ## Save arrays after preprocessing and windowing
            '''
             with open(os.path.join(args.output_path, "SubjectsVec.p"), 'wb') as outputFile:
            pickle.dump(all_subjects, outputFile)
            # Save the low activity indexes that was filtered out, this will use for the final validation of the model
            with open(os.path.join(args.output_path, "StdIndex.p"), 'wb') as outputFile:
                pickle.dump(StdIndex_all, outputFile)
            # Save the inclusion indices
            with open(os.path.join(args.output_path, "InclusionIndex.p"), 'wb') as outputFile:
                pickle.dump(inclusion_idx, outputFile)
            # Save the number of windows per subject
            with open(os.path.join(args.output_path, "NumWinSub.p"), 'wb') as outputFile:
                pickle.dump(NumWinSub, outputFile)
         
           '''
        # Save the data, labels and groups
        res = {'win_data_all_sub': win_data_all_sub,
               'win_labels_all_sub': win_labels_all_sub,
               'win_subjects': win_subjects,
               'win_chorea_all_sub': win_chorea_all_sub,
               'win_shift_all_sub': win_shift_all_sub,
               'win_video_time_all_sub': win_video_time_all_sub
               }
    
        if args.create_multi_class:
            res = preprocessing.get_label_chorea_comb(res)
            # res['arr_0'] = res['win_data_all_sub']
            # res['arr_1'] = res['win_labels_all_sub']
            # res['arr_2'] = res['win_subjects']
            # res['arr_5'] = res['win_chorea_all_sub']
            # res['arr_6'] = res['win_shift_all_sub']
            # res['arr_7'] = res['win_video_time_all_sub']
            np.savez(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model_{args.cohort}_only_{args.run_suffix}.npz'), **res)



if __name__ == '__main__':
    main()