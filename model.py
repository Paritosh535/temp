
# !pip install poppler-utils
# sudo yum install poppler-utils
# # sudo yum install poppler


# !pip install pycryptodome==3.0.0
# !pip install crypto
# %pip install poppler-utils
# %pip install layoutparser
# %pip install torchvision==0.8.0
# %pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
# %pip install "layoutparser[paddledetection]"
# %pip install "layoutparser[ocr]"

import layoutparser as sop
model_table = sop.Detectron2LayoutModel(
            config_path ='lp://TableBank/faster_rcnn_R_101_FPN_3x/config', # In model catalog
            label_map   ={0:"Table"}, # In model`label_map`
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.69] # Optional
        )
model_TT = sop.Detectron2LayoutModel(
            config_path ='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config', # In model catalog
            label_map   ={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}, # In model`label_map`
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.50] # Optional
        )
sop = sop
