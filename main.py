from tqdm import tqdm
from src.utils import *
import json
import argparse
import os
import sys

def model_extraction(input_path,output_path):
    ocr_df, pdf_images = load_ocr(input_path)
    json_pdf={}
    data_whole={}
    for _ in tqdm(range(len(pdf_images))):
        page_index = _
        image=np.asarray(pdf_images[page_index])
        image= clean_image(image,ocr_df,page_index)
        layout_df=layout_model(image,page_index)
        layout_df=correct_df_area(ocr_df,page_index,layout_df)
        data = extract_words(ocr_df,page_index,layout_df)
        data_whole[f'page_no_{_}']= data
    fname=input_path.replace(".pdf",".json").split('/')[-1]
    header=header_extraction(ocr_df)
    json_pdf["pages"]= data_whole
    json_pdf["header"]=header
    open(f"{output_path}{fname}" , "w", encoding="utf-8").write(json.dumps(json_pdf, indent=4, ensure_ascii=True))
    return f"{output_path}{fname}"

# Create the parser
parser = argparse.ArgumentParser(description='SOP')

# Add the arguments
parser.add_argument('--input_path',help='pdf file path')

parser.add_argument('--output_path',default="data/output/",help='pdf file path')
# Execute the parse_args() method
args = parser.parse_args()
print(args.output_path)
os.makedirs(args.output_path, exist_ok = True) 
print("Model Process Started..")
output=model_extraction(args.input_path,args.output_path)
print(f"Initial Json stored {output}")
