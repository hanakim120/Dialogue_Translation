import argparse
import openai
import json
import re
import logging
import os
import datetime
import time

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
    
def parseTranscript(subtitle_text, textdic, lang):
    for line in subtitle_text.strip().split('\n'):
        line = line.strip()
        if line.isnumeric():
            line_id = int(line)
        elif '-->' in line:
            continue
        else:
            line_text = line.replace('&lrm;', '')
            if line_id not in textdic:
                textdic[line_id] = dict()
            if lang in textdic[line_id]:
                textdic[line_id][lang] = textdic[line_id][lang] + " " + line_text.strip()
            else:
                textdic[line_id][lang] = line_text.strip()
    return textdic
                
def load_parser_and_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='chat-gpt', required=True)
    parser.add_argument('--data_mode', type=str, default='test')
    parser.add_argument('--debug_mode', action="store_true", help="use debug_mode")
    # path
    parser.add_argument('--output_dir', type=str, default='/home/hana/nas2/Dialogue_Translation/output/', required=True)
    parser.add_argument('--base_dir', type=str, default='/home/hana/nas2/Dialogue_Translation/')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--data_path', type=str, default='./data')
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
    parser.add_argument('--src_file_path',type=str)
    parser.add_argument('--trg_file_path',type=str)
    
    args = parser.parse_args()
    args.model_name_or_path = MODEL_MAPPING[args.model_type]

    return parser, args