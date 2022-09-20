# Neural Style Transfer 

Author: Danis Alukaev <br>
Email: d.alukaev@innopolis.univeristy <br>
Group: B19-DS-01

This directory contains deliverables for Assignment 1 on Practical Machine Learning & Deep Learning course:
1. Compeleted version of a given ipynb file
2. Python script `nst_generate.py`
3. Content, style, and generated images
4. Video `demo.mp4` generated from the `model_nn` saved interim images

Supplementary files are:

5. File with necessary dependencies (some of them are very sensitive)
6. This `README.md` with manual

### How to use?
1. Create new conda environment with Python of version 3.7
```
conda create -n pmldl_a1 python=3.7
```
2. Activate the environment
```
conda activate pmldl_a1
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. In case you want to run algorithm in JupyterLab:
```
cd <WORKING_DIRECTORY_WITH_DELIVERABLES>
jupyter lab
```
5. In case you want to run Python script:
```
python nst_generate.py <PATH_CONTENT_IMAGE> <PATH_STYLE_IMAGE> <SAVING_PATH>
```
Example of usage:
```
python nst_generate.py content.jpg style.jpg generated.jpg
```
**IMPORTANT NOTE**: resolution of content and style images should be the same and equal to `300x400` pixels. It was set in `nst_utils` by author, which we cannot modify by specification, so I have decided to leave it as it is.

**YET ANOTHER IMPORTANT NOTE**: for each experiment program creates a new directory `output/<DATETIME>`, where it will be saving interim images.

### Results

I have prepared two videos with demonstration of style transfering for a [photo of Louvre museum with a painting by Claude Monet](https://youtu.be/FAXe5g5hFEU) and [Audrey Hepburn portrait with Cubism art](https://youtu.be/Ngszk3q_QQ4). Check it out!