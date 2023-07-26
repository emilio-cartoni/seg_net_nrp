Prednet
=======

Installation:
-------------

- Create new conda environment: `conda env create --file environment.yml`
- In conda environment, open file: `<PREDNET_ENV>/lib/python3.6/site-packages/keras/engine/saving.py`
    - In file, erase all mentions of:
        - `.encode('utf8')`
        - `.encode('utf-8')`
        - `.decode('utf8')`
        - `.decode('utf-8')`

Running:
--------

- In `create_mask.py`, set parameters:
    - `nt`: number of images in sequence
    - `data_dir`: Folder pointing to a sequence of images
    - `out_dir`: Folder to save masks to
- Activate conda environment
- Run `python create_mask.py`