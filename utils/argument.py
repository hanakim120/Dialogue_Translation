import argparse
import openai
import json
import re
import logging
import os
import datetime
import time
from tqdm import tqdm

MODEL_MAPPING = {
    'text-davinci-003': 'text-davinci-003',
    'text-curie-001': 'text-curie-001',
    'chat-gpt': 'gpt-3.5-turbo'
}

def openai_login(args):
    with open('./utils/key.json') as f:
        keys = json.load(f)
    if args.key_org == "tutoring":
        openai.organization = keys[0] 
    openai.api_key = keys[1] 

def read_json(filename):
    f = open(f'{filename}', 'r',encoding='utf-8-sig')
    data = []
    for line in f.readlines():
        dic = json.loads(line)
        data.append(dic)
    return data

# set logger

logger = logging.getLogger('logger')

    
def init_logger(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        
    # log file 로 저장
    handler = logging.FileHandler(os.path.join(args.log_dir, '{:%Y-%m-%d-%H:%M:%S}.log'.format(datetime.datetime.now())), encoding='utf=8')
    logger.addHandler(handler)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.log_level in [-1, 0] else logging.WARN,
    )
    logger.warning(args)
    
# def parseTranscript(subtitle_text, textdic, lang):
#     for line in subtitle_text.strip().split('\n'):
#         line = line.strip()
#         if line.isnumeric():
#             line_id = int(line)
#         elif '-->' in line:
#             continue
#         else:
#             line_text = line.replace('&lrm;', '')
#             if line_id not in textdic:
#                 textdic[line_id] = dict()
#             if lang in textdic[line_id]:
#                 textdic[line_id][lang] = textdic[line_id][lang] + " " + line_text.strip()
#             else:
#                 textdic[line_id][lang] = line_text.strip()
#     return textdic

def parseTranscript(subtitle_text, textdic):
    
    line_list = []
    for line in subtitle_text.strip().split('\n'):
        line = line.strip()
        if line.isnumeric():
            line_id = int(line)
        elif '-->' in line:
            continue
        else:
            line_text = line.replace('&lrm;', '')
            line_list.append(line_text)
            # if line_id not in textdic:
            #     textdic[line_id] = dict()
            # if lang in textdic[line_id]:
            #     textdic[line_id][lang] = textdic[line_id][lang] + " " + line_text.strip()
            # else:
            #     textdic[line_id][lang] = line_text.strip()
    return line_list


def prepro_netflix_manual_session(source_content, target_content, args) :  
    
    # 대화 정보
    dialogue = source_content

    # 화자 정보 # 임시
    speakers = ['동은', '연진', '동은', '연진', '동은', '연진', '동은', '연진', '동은', '연진', '동은', '연진']  # TODO speaker

    # 장면 정보 # 임시
    scenes_label = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2] # TODO scene anotation

    # 출력할 결과 리스트
    result = []

    # 각 장면별 대화 추출 및 변환
    for scene_id in range(max(scenes_label) + 1):
        # 장면에 대한 대화 인덱스 리스트
        scene_dialogue_indices = [i for i, label in enumerate(scenes_label) if label == scene_id]

        # 장면 대화 리스트
        scene_dialogue = [dialogue[i] for i in scene_dialogue_indices]
        scene_speakers = [speakers[i] for i in scene_dialogue_indices]


        dialogue_info_list = []
        for i, d in enumerate(scene_dialogue):
            # 대화 정보 생성
            dialog_info = {
                'src_lang': 'en',
                'trg_lang': 'ko',
                'text': d,
                'speaker': scene_speakers[i],
                'label': ""
            }

            dialogue_info_list.append(dialog_info)

        # 장면 정보 생성
        scene_info = {
            'id': f'interactive-Scenes_label_{scene_id}-EP1', # TODO episode hard coding 
            'dialog': dialogue_info_list
        }

        # 결과 리스트에 장면 정보 추가
        result.append(scene_info)
        
    return result
    
    
def prepro_netflix_auto_session(source_content, target_content, args):

    manual_split_num = args.manual_split_num
    dialogue = source_content
    trg_dialogue = target_content

    speakers = [] # TODO speaker

    scenes = []
    for i in range(0, len(dialogue), manual_split_num):
        scene_dialogue = dialogue[i:i+manual_split_num]
        scene_speakers = [] # speakers[i:i+manual_split_num]
        scene_dialogue_target = trg_dialogue[i:i+manual_split_num]
        scene = {'id': f'interactive-scene-{i//manual_split_num}-EP1', 'dialog': []}  # TODO episode hard coding 
        for j in range(len(scene_dialogue)):
            dialog = {
                'src_lang': 'en',
                'trg_lang': 'ko',
                'text': scene_dialogue[j],
                'speaker': '', # scene_speakers[j]
                'label': scene_dialogue_target[j]
            }
            scene['dialog'].append(dialog)
        scenes.append(scene)
    
    return scenes

def prepare_utter_level_input_for_gpt_model(args):
    # gpt input : dialogue context + current turn (utter-level)
    
    with open(args.data_path + args.data_dir,'r') as f:
        data = json.load(f)
    
    # Attach a name tag
    whole_dialog = []  
    
    for key_id in data.keys():
        dialog_w_name = []
        for utter in data[key_id]:
            input_dialog = utter['source']
            if utter['speaker']=='agent':
                name_tag_A = 'Agent'
                dialog_w_name.append(name_tag_A + ': ' + input_dialog.strip())
            else : 
                name_tag_B = 'Customer'
                dialog_w_name.append(name_tag_B + ': ' + input_dialog.strip())
            
        whole_dialog.append(dialog_w_name) 
        
    cur_text_dataset = []
    label_dataset = []
    input_history = []
    input_history_list = []
    
    for dialog in whole_dialog:
        history = []
        for t_idx, turn in enumerate(dialog[::2]):
            idx = t_idx *2 # 0 2 4 6 8 ... 
            if len(dialog) % 2 != 0 :
                if idx != len(dialog)-1:
                    label_response = dialog[idx+1] # 1,3, 5, 7, 9, 11, 13
                cur_text = turn 
            else : 
                label_response = dialog[idx+1] # 1,3, 5, 7, 9, 11, 13, 15
                
            cur_text = turn + '\n' + label_response
            history += [cur_text]
            
            cur_text_dataset.append(cur_text)
            input_history.append("\n".join(history))
            label_dataset.append(label_response)
            
        input_history_list.append(history)

    return cur_text_dataset, label_dataset, input_history, input_history_list

def prepare_dialog_level_input_for_gpt_model(args):
    # gpt input : dialog-level (whole dialog)
    
    whole_dialog = []
    src_trg_pair = []

    with open(args.data_path + args.data_dir,'r') as f:
        data = json.load(f)
        
    # Attach a name tag and make dialog-level src-trg pair data      
    for dialog_idx, key_id in tqdm(enumerate(data.keys())):
        dialog_w_name = []
        label_dialog_list =[]
        for utter in data[key_id]:
            input_dialog = utter['source']
            label_dialog = utter['target']
            
            if utter['speaker']=='agent':
                name_tag_A = 'Agent'
                dialog_w_name.append(name_tag_A + ': ' + input_dialog.strip())
            else : 
                name_tag_B = 'Customer'
                dialog_w_name.append(name_tag_B + ': ' + input_dialog.strip())
            label_dialog_list.append(label_dialog)
            
        whole_dialog.append(dialog_w_name) 
        
        src_trg_pair.append({
                'dialog_index': dialog_idx,
                'source' : dialog_w_name,
                'target_label' : label_dialog_list,
            })
        
    # save the src-trg pair data     
    file_path = os.path.join(args.data_path, f"{args.memo}_src_trg_pair.json")
    with open(file_path, 'w') as f:
        json.dump(src_trg_pair, f, indent=2)
    
    return whole_dialog

def save_log_and_result(epi_idx, start_index, divide_index, context_ws, pred_A_summ_list, pred_B_summ_list, results):
    
    # log 파일 저장
    logger.info("  episode_idx : {}".format(epi_idx))
    logger.info("  data_idx : {}".format(start_index))
    logger.info("  divide_index': {}".format(divide_index))
    logger.info("  pred_A_summ : {}".format(pred_A_summ_list))
    logger.info("  pred_B_summ : {}".format(pred_B_summ_list))
    logger.info("")
    
    results.append({
        'episode_index': epi_idx,
        'data_index': start_index,
        'divide_index': divide_index,
        'dialog_context' : context_ws, # string
        'pred_A_summ': pred_A_summ_list, # list
        'pred_B_summ': pred_B_summ_list # list
    })
    return results


def load_parser_and_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='chat-gpt', required=True)
    parser.add_argument('--debug_mode', action="store_true", help="use debug_mode")
    # path
    parser.add_argument('--output_dir', type=str, default='/home/hana/nas2/Dialogue_Translation/output/', required=True)
    parser.add_argument('--base_dir', type=str, default='/home/hana/nas2/Dialogue_Translation/')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--prompt_dir', type=str, default='./prompts')
    parser.add_argument('--log_level', type=int, default=-1)
    parser.add_argument('--max_tokens', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--stop_seq', type=str, default='\n')
    parser.add_argument('--key_org', type=str, default='tutoring')
    parser.add_argument('--memo', type=str)
    parser.add_argument('--source_lang',type=str, default='English')
    parser.add_argument('--target_lang',type=str, default='Korean')
    parser.add_argument('--src_file_path',type=str)
    parser.add_argument('--trg_file_path',type=str)
    parser.add_argument('--use_manual_session',action='store_true')
    parser.add_argument('--input_type',choices=['whole_dialog','context_and_current_turn'])
    parser.add_argument('--manual_split_num',type=int, default=15)
    parser.add_argument('--persona_ws',type=int, default=10)
    parser.add_argument('--extract_persona_type',choices=['dialog-level','utter-level'])
    parser.add_argument('--dialog_type', choices=['conversation','movie_script'])
    parser.add_argument('--merged_persona_file', type=str)
    args = parser.parse_args()
    args.model_name_or_path = MODEL_MAPPING[args.model_type]

    return parser, args