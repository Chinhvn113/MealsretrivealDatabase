First clone RAM plus plus repo:
```bash
git clone https://github.com/xinyu1205/recognize-anything.git
cd recognize-anything
pip install -e .
```
After that install database library
```bash
cd MealsretrievalDatabase
pip install -e .
```
Simple use:
```python
from database import FAISSmanager, set_all_seed, build_transform, dynamic_preprocess, load_image
set_all_seed(42)
faissmanager = FAISSmanager()
faissmanager.build('data/root')
result = faissmanager.search(query_type = 'text', query ='hello world')
```


