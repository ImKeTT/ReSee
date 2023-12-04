python3 dialogpt2.py --dataset wow --max_length 200 --model_name microsoft/DialoGPT-medium --load microsoft-DialoGPT-medium_wow_trunc &&\
python3 dialogpt2.py --dataset dd --max_length 150 --model_name microsoft/DialoGPT-medium --load microsoft-DialoGPT-medium_dd_trunc &&\
python3 dialogpt2.py --dataset wow --max_length 200 --model_name gpt2-medium --load gpt2-medium_wow_trunc &&\
python3 dialogpt2.py --dataset dd --max_length 150 --model_name gpt2-medium --load gpt2-medium_dd_trunc &&\
python3 dialogpt2.py --dataset wow --max_length 200 --model_name microsoft/DialoGPT-medium --no_finetune --load microsoft-DialoGPT-medium_wow_trunc &&\
python3 dialogpt2.py --dataset dd --max_length 150 --model_name microsoft/DialoGPT-medium --no_finetune --load microsoft-DialoGPT-medium_dd_trunc &&\
python3 dialogpt2.py --dataset wow --max_length 200 --model_name gpt2-medium --no_finetune --load gpt2-medium_wow_trunc &&\
python3 dialogpt2.py --dataset dd --max_length 150 --model_name gpt2-medium --no_finetune --load gpt2-medium_dd_trunc