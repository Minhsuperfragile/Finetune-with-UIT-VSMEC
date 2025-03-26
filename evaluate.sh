# Evaluate checkpoint model, change folder name to yours

#python main.py --model ./vietnamese-bi-encoder-25-3-16-00 --task evstd --config ./config/vietnamese-bi-encoder.json --save-folder vietnamese-bi-encoder-25-3-16-40
#python main.py --model ./gte-multilingual-mlm-base-25-3-17-54 --task evstd --config ./config/gte-multilingual-mlm-base.json --save-folder gte-multilingual-mlm-base-25-3-17-54
python main.py --model gte-multilingual-mlm-base-26-3-6-7 --task evsim --confif ./config/gte-multilingual-mlm-base.json --save-folder gte-multilingual-mlm-base-26-3-6-7