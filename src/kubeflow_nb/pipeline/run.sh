cd scripts
python3 dataloader.py
python3 preprocess.py
python3 modeling.py

cd ..
python3 pipeline.py
python3 main.py