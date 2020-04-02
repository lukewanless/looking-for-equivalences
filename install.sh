pip3 install torch torchvision
pip3 install pandas
pip3 install sklearn
pip3 install transformers
pip3 install spacy
pip3 install nltk
pip3 install xgboost
pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
echo -e "import nltk\nnltk.download('wordnet')" > w_download.py
python3 w_download.py
rm w_download.py
