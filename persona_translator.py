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


    # load data
    if args.extract_persona_type == 'dialog-level':
        input_history_list = prepare_dialog_level_input_for_gpt_model(args)

    elif args.extract_persona_type == 'utter-level': # turn
        cur_text_dataset, label_dataset, input_history, input_history_list = prepare_utter_level_input_for_gpt_model(args)
        
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
    divide_index = 0
    ws = args.persona_ws
    
    if args.extract_persona_type == 'dialog-level':
        
        if args.dialog_type == "conversation" :
            
            for epi_idx, whole_dialog in enumerate(input_history_list): # 대화수
            
                model_input = prompt % (args.source_lang, args.target_lang, whole_dialog)   
                pred_string = model.chatgpt_inference(model_input,args.debug_mode) 
            
                # log 파일 저장
                logger.info("  data_idx : {}".format(start_index))
                logger.info("  source : {}".format(whole_dialog))
                logger.info("  target : {}".format(pred_string))
                logger.info("")
                
                predict.append({
                    'data_index': start_index,
                    'source' : whole_dialog,
                    'predicted_target' : pred_string,
                })
                        
                # save the results
                with open(os.path.join(args.output_dir, 'results_{}_{}.json'.format(args.source_lang,args.memo)), 'w', encoding='utf-8') as f:
                    json.dump(predict, f, indent=4,ensure_ascii=False)    
            
                start_index += 1
            
            
        elif args.dialog_type == "movie_script": # TODO
            divide_index = 0
            ws = args.persona_ws
        
    elif args.extract_persona_type == "utter-level":
        
        if args.dialog_type == "conversation":
            
            for idx_input_history, input_history_turn_list in enumerate(input_history_list): # 대화수
                for idx_data, data in tqdm(enumerate(input_history_turn_list)): # idx_data 0~5
                    
                    current_turn = data
                    
                    if idx_data == 0 :
                        dialog_context = ""
                        whole_dialog_input = dialog_context + '\n' + current_turn
                    else :
                        dialog_context = input_history_turn_list[:idx_data]

                        whole_dialog_input = dialog_context[0] + '\n' + current_turn
                    
                    # put the data into the prompt
                    dialog_context_input = dialog_context
                    model_input = prompt % (args.source_lang, args.target_lang, whole_dialog_input)

                    pred_string = model.chatgpt_inference(model_input,args.debug_mode) 
                    
                    # log 파일 저장
                    logger.info("  data_idx : {}".format(start_index))
                    logger.info("  source : {}".format(whole_dialog_input))
                    logger.info("  target : {}".format(pred_string))
                    logger.info("")
                    
                    predict.append({
                        'data_index': start_index,
                        'source' : whole_dialog_input,
                        'predicted_target' : pred_string,
                    })
                            
                    # save the results
                    with open(os.path.join(args.output_dir, 'results_{}_{}.json'.format(args.source_lang,args.memo)), 'w', encoding='utf-8') as f:
                        json.dump(predict, f, indent=4,ensure_ascii=False)    
                
                    start_index += 1
                    
        elif args.input_type == "movie_script": 
            pass # TODO
        
        
if __name__ == "__main__":
    parser, args = load_parser_and_args()
    main(args)