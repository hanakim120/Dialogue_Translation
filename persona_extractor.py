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
        
        return output_string
    
def main(args):  
    # make output file
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
    
    preds_A = []; preds_B = []; results = []
    
    generate_fail = 0
    start_index = 0
    divide_index = 0
    ws = args.persona_ws
    
    if args.extract_persona_type == 'dialog-level':
        
        if args.dialog_type == "conversation" :
            
            for epi_idx, whole_dialog in enumerate(input_history_list): # 대화수
            
                model_input = prompt % (whole_dialog)   
                pred_string = model.chatgpt_inference(model_input,args.debug_mode) 
            
                try:
                    # postprocessing
                    pred_A_summ = pred_string.split("\nCustomer's persona sentences:")[0].split("Agent's persona sentences:")[1]
                    pred_B_summ = pred_string.split("\nCustomer's persona sentences:")[1]
                
                    _pred_A_summ_list = [pred_A_summ]
                    _pred_B_summ_list = [pred_B_summ]
                    
                    pred_A_summ_list = [s.strip("' ").strip('" ') for s in _pred_A_summ_list]
                    pred_B_summ_list = [s.strip("' ").strip('" ') for s in _pred_B_summ_list]
                    
                    # log & result 저장 : dialog-level conversation 은 divide_index 항상 0
                    results = save_log_and_result(epi_idx, start_index, divide_index, whole_dialog, pred_A_summ_list, pred_B_summ_list, results)
                    
                    preds_A.append(pred_A_summ_list) 
                    preds_B.append(pred_B_summ_list) 
                        
                # 생성 잘 안된 경우 예외 처리
                except:
                    generate_fail += 1
                    logger.info("  generate_fail idx : {}".format(start_index)) # save the fail index
                    logger.info("  data_idx : {}".format(start_index))
                    logger.info("  divide_index': {}".format(divide_index))
                    logger.info("  fail_pred_string : {}".format(pred_string))
                    logger.info("")
                    
                    results.append({
                        'episode_index': epi_idx,
                        'fail_index': start_index,
                        'divide_index': divide_index,
                        'dialog_context' : whole_dialog, 
                        'fail_pred_string': pred_string 
                    })
                    pass
            
                # save the results
                with open(os.path.join(args.output_dir, 'results_{}.json'.format(args.memo)), 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4)  

                start_index += 1  

            logger.info("Compute metrics to: {}".format(args.output_dir))
            logger.info("Generate fail count : {}".format(generate_fail))   
            
        elif args.dialog_type == "movie_script": # TODO
            
            for idx_input_history, input_history_turn_list in enumerate(input_history_list): # 대화수
                iter = len(input_history_turn_list)//ws + 1 
                divide_index = 0
                for i in range(iter):
                    try:
                        context_ws = input_history_turn_list[i*ws:i*ws+ws+1]
                    except:
                        context_ws = input_history_turn_list[i*ws:]
                        model_input = prompt % (context_ws)
                        
                    pred_string = model.chatgpt_inference(model_input,args.debug_mode) 
                
                    try:
                        # postprocessing
                        pred_A_summ = pred_string.split("\nStudent's persona sentences:")[0].split("Teacher's persona sentences: ")[1] # string
                        pred_B_summ = pred_string.split("\nStudent's persona sentences:")[1] # string
                        
                        _pred_A_summ_list = [pred_A_summ]
                        _pred_B_summ_list = [pred_B_summ]
                        
                        pred_A_summ_list = [s.strip("' ").strip('" ') for s in _pred_A_summ_list]
                        pred_B_summ_list = [s.strip("' ").strip('" ') for s in _pred_B_summ_list]
                        
                        # log & result 저장
                        results = save_log_and_result(epi_idx, start_index, divide_index, context_ws, pred_A_summ_list, pred_B_summ_list, results)     

                        preds_A.append(pred_A_summ_list) 
                        preds_B.append(pred_B_summ_list) 
                            
                    
                    # 생성 잘 안된 경우 제외
                    except:
                        generate_fail += 1
                        logger.info("   generate_fail idx : {}".format(start_index)) # save the fail index
                        logger.info("  data_idx : {}".format(start_index))
                        logger.info("  divide_index': {}".format(divide_index))
                        logger.info("  fail_pred_string : {}".format(pred_string))
                        logger.info("")
                        
                        results.append({
                            'episode_index': idx_input_history,
                            'fail_index': start_index,
                            'divide_index': divide_index,
                            'dialog_context' : context_ws, # string
                            'fail_pred_string': pred_string # list
                        })
                        pass
                        
                    # save the results
                    with open(os.path.join(args.output_dir, 'results_{}.json'.format(args.memo)), 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=4)   
                            
                    divide_index += 1
                start_index += 1
                
            logger.info("Compute metrics to: {}".format(args.output_dir))
            logger.info("Generate fail count : {}".format(generate_fail))   
        
    elif args.extract_persona_type == "utter-level":
        
        if args.dialog_type == "conversation":
            
            for idx_input_history, input_history_turn_list in enumerate(input_history_list): # 대화수
                for idx_data, data in tqdm(enumerate(input_history_turn_list)): # idx_data 0~5
                    
                    current_turn = data
                    
                    if idx_data == 0 :
                        dialog_context = ""
                    else :
                        dialog_context = input_history_turn_list[:idx_data]
            
                    # put the data into the prompt
                    if len(dialog_context) < ws:
                        dialog_context_input = dialog_context
                        model_input = prompt % (dialog_context_input,current_turn)
                    else :
                        dialog_context_input = dialog_context[-ws:]
                        model_input = prompt % (dialog_context_input,current_turn)
                        
                    pred_string = model.chatgpt_inference(model_input,args.debug_mode) 
                    
                    try:
                        
                        # postprocessing
                        pred_A_summ = pred_string.split("\nCustomer's persona sentences:")[0].split("Agent's persona sentences:")[1] # string
                        pred_B_summ = pred_string.split("\nCustomer's persona sentences:")[1] # string
                        
                        _pred_A_summ_list = [pred_A_summ]
                        _pred_B_summ_list = [pred_B_summ]
                        
                        pred_A_summ_list = [s.strip("' ").strip('" ') for s in _pred_A_summ_list]
                        pred_B_summ_list = [s.strip("' ").strip('" ') for s in _pred_B_summ_list]
                        
                        logger.info("  episode_idx : {}".format(idx_input_history))
                        logger.info("  data_idx : {}".format(start_index))
                        logger.info("  divide_index': {}".format(divide_index))
                        logger.info("  pred_A_summ : {}".format(pred_A_summ_list))
                        logger.info("  pred_B_summ : {}".format(pred_B_summ_list))
                        logger.info("")
                        
                        results.append({
                            'episode_index': idx_input_history,
                            'data_index': start_index,
                            'divide_index': divide_index,
                            'dialog_context' : dialog_context, # string
                            'current_turn' : current_turn,
                            'pred_A_summ': pred_A_summ_list, # list
                            'pred_B_summ': pred_B_summ_list # list
                        })
                        
                        preds_A.append(pred_A_summ_list) 
                        preds_B.append(pred_B_summ_list) 
                
                    # 생성 잘 안된 경우 제외
                    except:
                        generate_fail += 1
                        logger.info("   generate_fail idx : {}".format(start_index)) # save the fail index
                        logger.info("  data_idx : {}".format(start_index))
                        logger.info("  divide_index': {}".format(divide_index))
                        logger.info("  fail_pred_string : {}".format(pred_string))
                        logger.info("")
                        
                        results.append({
                            'episode_index': idx_input_history,
                            'fail_index': start_index,
                            'divide_index': divide_index,
                            'dialog_context' : dialog_context, # string
                            'current_turn' : current_turn,
                            'fail_pred_string': pred_string # list
                        })
                        pass 
                    
                    # save the results
                    with open(os.path.join(args.output_dir, 'results_{}.json'.format(args.memo)), 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=4) 
                        
                    start_index += 1
                    
            logger.info("Compute metrics to: {}".format(args.output_dir))
            logger.info("Generate fail count : {}".format(generate_fail)) 
            
        elif args.input_type == "movie_script": 
            pass # TODO
        
        
if __name__ == "__main__":
    parser, args = load_parser_and_args()
    main(args)
    
