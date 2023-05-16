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
import pdb
import pandas as pd
from icecream import ic
from utils.argument import load_parser_and_args, openai_login, read_json, init_logger, parseTranscript, prepro_netflix_manual_session, prepro_netflix_auto_session, prepare_utter_level_input_for_gpt_model, prepare_dialog_level_input_for_gpt_model, save_log_and_result

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
                print("** gpt-4 inference start **")
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
        #ic(output)
        return output_string
    
def main(args):   
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup logging
    init_logger(args)

    #src_file = args.src_file_path
    trg_file = args.trg_file_path

    #with open(src_file, 'r') as f:
        #source_content = f.read()

    with open(trg_file, 'r') as f2:
        target_content = f2.read()

    textdic = dict()
    
    # load data
    #input_history_list_src = parseTranscript(src_file, textdic)
    #input_history_list = parseTranscript(target_content, textdic)

    # read prompt
    with open(args.prompt_dir, 'r') as f:
        prompt = f.read()
            
    logger = logging.getLogger('logger')        
    logger.info("***** Prompt *****")
    logger.info(prompt) 
    
    # load model
    model = GPT3(args)
    
    # openai login 
    openai_login(args) 
    
    source = [] ; predict = []
    
    generate_fail = 0
    start_index = 0
    ws = args.persona_ws
    
    target_list = target_content.split('\n\n')
    #import pdb; pdb.set_trace()
    
    if len(target_list) > ws:
        div = len(target_list)//ws
    
        for i in range(div):
            start_index = i * ws
            end_index = start_index + ws
            
            try:
                context_ws = target_list[start_index:end_index]
            except:
                context_ws = target_list[start_index:]
            
            model_input = prompt % (context_ws)
            pred_string = model.chatgpt_inference(model_input,args.debug_mode) 

            #log 파일 저장
            #logger.info("  data_idx : {}".format(start_index))
            logger.info("  divide_idx : {}".format(i))
            logger.info("  before : {}".format(context_ws))
            logger.info("  predict : {}".format(pred_string))
            logger.info("")
            
            predict.append({
                'divide_index': i,
                'divide_idx' : context_ws,
                'predict' : pred_string
            })

                    
            # save the results
            with open(os.path.join(args.output_dir, 'results_{}_{}.json'.format(args.source_lang,args.memo)), 'w', encoding='utf-8') as f:
                json.dump(predict, f, indent=4,ensure_ascii=False)    
        
            
    else : 
        context_ws = target_content
        
        model_input = prompt % (context_ws)
        pred_string = model.chatgpt_inference(model_input,args.debug_mode) 

        #log 파일 저장
        #logger.info("  data_idx : {}".format(start_index))
        logger.info("  divide_idx : {}".format(0))
        logger.info("  before : {}".format(context_ws))
        logger.info("  predict : {}".format(pred_string))
        logger.info("")
        
        predict.append({
            'divide_index': 0,
            'divide_idx' : context_ws,
            'predict' : pred_string
        })

                
        # save the results
        with open(os.path.join(args.output_dir, 'results_{}_{}.json'.format(args.source_lang,args.memo)), 'w', encoding='utf-8') as f:
            json.dump(predict, f, indent=4,ensure_ascii=False)    
        
        
if __name__ == "__main__":
    parser, args = load_parser_and_args()
    main(args)