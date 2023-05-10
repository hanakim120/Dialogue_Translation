import sys
import os
import json
import math
import glob
import ast
import logging
import time
import random
import pickle
import shutil
import pathlib
import numpy as np
import openai
from tqdm import tqdm
from collections import defaultdict
import pdb
from icecream import ic
from utils.argument import load_parser_and_args, openai_login, read_json, init_logger, parseTranscript
 
class GPT3(object):
    def __init__(self, args):
        self.model_name = args.model_name_or_path
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.frequency_penalty = args.frequency_penalty
        self.presence_penalty = args.presence_penalty
        self.stop_seq = args.stop_seq
        self.debug_mode = args.debug_mode
        self.org_key = 0
        self.private_key = 1

    def chatgpt_inference(self, prompt, debug_mode=False):
        while True:
            try:
                print("** gpt-3.5-turbo inference start **")
                output = openai.ChatCompletion.create(                    
                    model=self.model_name,
                    messages=[
                            {"role": "user", "content": prompt},
                        ],
                    n=1, 
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=self.stop_seq
                )
                break
            
            # connection error .. etc.. 
            except Exception as e:
                print(e) 
                time.sleep(15)
                
        if debug_mode:
            return output

        output_string = output['choices'][0]['message']['content'] 
        output_string = output_string.strip() 
        ic(output)
        return output_string
    
def main(args):   
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
 
    # Setup logging
    init_logger(args)

    # load_data
    src_file = args.src_file_path
    trg_file = args.trg_file_path
    
    with open(src_file, 'r') as f:
        source_content = f.read()

    with open(trg_file, 'r') as f2:
        target_content = f2.read()

    textdic = dict()
    
    textdic = parseTranscript(source_content, textdic, args.source_lang)
    
    # read prompt
    with open(args.prompt_dir, 'r') as f:
        prompt = f.read()
        

 
    # load model
    model = GPT3(args)
    
    # openai login 
    openai_login(args) 
    
    source = []; predict = []
    
    generate_fail = 0
    start_index = 0
    
    logger = logging.getLogger('logger')        
    logger.info("***** Prompt *****")
    logger.info(prompt) 
    
    for i in textdic.keys():
        input_line = textdic[i][args.source_lang]
        # put the data into the prompt
        model_input = prompt % (args.source_lang, input_line)
        #logger.info(model_input) 
        pred_string = model.chatgpt_inference(model_input,args.debug_mode) 
            
        # log 파일 저장
        logger.info("  data_idx : {}".format(start_index))
        logger.info("  source : {}".format(input_line))
        logger.info("  target : {}".format(pred_string))
        logger.info("")
        
        predict.append({
            'data_index': start_index,
            'source' : input_line,
            'predicted_target' : pred_string,
        })
                
        # save the results
        with open(os.path.join(args.output_dir, 'results_{}_{}.json'.format(args.source_lang,args.memo)), 'w', encoding='utf-8') as f:
            json.dump(predict, f, indent=4,ensure_ascii=False)    
    
        start_index += 1
            

if __name__ == "__main__":
    parser, args = load_parser_and_args()
    main(args)