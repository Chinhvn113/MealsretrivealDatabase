First clone RAM plus plus repo:
```bash
git clone https://github.com/xinyu1205/recognize-anything.git
cd recognize-anything
pip install -e .
```
After that install database library
```bash
cd ..
cd MealsretrivevalDatabase
pip install -e .
```
Download the weights:
```bash
apt-get install git-lfs
git lfs install
git clone https://huggingface.co/xinyu1205/recognize-anything-plus-model
```
Download test keyframes
```bash
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/1LKZHyhjhLGnbQHXXBN7CjrFD2npN95mw/view
unzip toidayyy.zip
```
Simple use:
```python
from database import FAISSManager, set_all_seeds, build_transform, dynamic_preprocess, load_image
set_all_seeds(42)
faissmanager = FAISSManager()
faissmanager.build('data/root')
result = faissmanager.search(query_type = 'text', query ='hello world')
```


