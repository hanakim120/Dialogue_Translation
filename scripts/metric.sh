# baseline dialog-level 
python ./metric.py --src_file ./output/persona_translator/whole_dialog_0shot/results_English_BConTrasT_test_30episode.json --ref_file ./data/BConTrasT/prepro/BConTrasT_test30_whole_src_tag_pair.json --output_dir ./output/persona_translator/metrics/

# ours dialog-level 
python ./metric.py --src_file ./output/persona_translator/whole_dialog_0shot_ours/results_English_BConTrasT_test_30episode_prompt_2.json --ref_file ./data/BConTrasT/prepro/BConTrasT_test30_whole_src_tag_pair.json --output_dir ./output/persona_translator/metrics/