# Select one command to run 

# train mGTE model with constrastive task
python main.py --model gte --task ftsim --save-folder gteSimFT
# train vietnamese-bi-encoder model with contrastive task
python main.py --model vbe --task ftsim 

# train mGTE model with standard classification task
python main.py --model gte --task ftstd
# train vietnamese-bi-encoder model with standard classification task
python main.py --model vbe --task ftstd

# train Qwen2.5 with standard 
#TODO