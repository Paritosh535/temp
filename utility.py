import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import layoutparser as sop
from src.model import model_TT, model_table, sop

def extract_line(df_ocr, df_area):
    data=[]
    bold=[]
    for idn,model_row in df_area.iterrows():
        tp_bbx=[0,int(model_row.y_1),int(model_row.page_width),int(model_row.y_2)]
        words= []
        is_bold = []
        for idx,ocr_row in df_ocr.iloc[:,:4].iterrows():
            word_bbx=ocr_row.astype(int).values
            flag=check_point(tp_bbx,word_bbx)
            if flag:
                words.append(df_ocr.iloc[idx]['text'].strip())
                is_bold.append(df_ocr.iloc[idx]['is_bold'])
        bold.append(is_bold)
        data.append(words)
    return data,bold


def check_title(df):
    final=[]
    for i,row in df.iterrows():
        if row.type in ['Title','text']:
            if (row.is_upper==True) and (row.word_length<5): # and (row.is_bold==True):
                final.append("Title")
            else:
                final.append("Text")
        else:
             final.append(row.type)
    df["final_type"] = final
    return df

def is_header_footer(ocr_df):
    ocr_df["combine_info"]= ocr_df[['x_1','y_1','x_2','y_2','text','height','width']].apply(lambda x: ','.join([str(p) for p in x]) ,axis=1)
    ocr_df["is_header_footer"]=ocr_df.combine_info.duplicated()
    return ocr_df

def is_bulit(ocr_df):
    title_index=["1.0",'2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0']
    ocr_df['is_start_bulit']=ocr_df.text.apply(lambda x : x in title_index)
    return ocr_df

# word
def get_center(x1,y1,x2,y2):
    xCenter = (x1 + x2) / 2
    yCenter = (y1 + y2) / 2
    return xCenter,yCenter

def FindPoint(x1, y1, x2,
              y2, x, y) :
    if (x > x1 and x < x2 and
        y > y1 and y < y2) :
        return True
    else :
        return False

def NMS(boxes, overlapThresh = 0.70):
    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts
    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        # Create temporary indices  
        temp_indices = indices[indices!=i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
        if np.any(overlap) > 0.4:
            indices = indices[indices != i]
    print(indices)
    #return only the boxes at the remaining indices
#     return boxes[indices]
    return indices.tolist()

def check_point(tt_box,word_bbx):
    point_x,point_y = get_center(word_bbx[0],word_bbx[1],
                                 word_bbx[2],word_bbx[3])
    return FindPoint(tt_box[0],tt_box[1],
                     tt_box[2],tt_box[3],point_x,point_y)

def extract_words(df_ocr, page_index, layout_df):
    df_ocr=df_ocr[df_ocr.page_index==page_index]
    df_area = layout_df.sort_values(by=['y_1'], ascending=True)
    data=[]
    for idn,model_row in df_area.iterrows():
        tp_bbx=[0,int(model_row.y_1),int(model_row.page_width),int(model_row.y_2)] #model_row[:4].astype(int).values
        para_data = {}
        word_index=[]
        words= []
        for idx,ocr_row in df_ocr.iloc[:,:4].iterrows():
            word_bbx=ocr_row.astype(int).values
            flag=check_point(tp_bbx,word_bbx)
            if flag:
                word_index.append(idx)
                words.append(df_ocr.iloc[idx]['text'])
                
        para_data["page_index"]=model_row.page_index
        para_data["type"]=model_row.type
        para_data["final_type"]=model_row.final_type
#         para_data['words'] = words
        content=" ".join([df_ocr.iloc[x]['text'] for x in word_index])
#         strencode = content.encode("ascii", "ignore")
#         #decode() method
#         strdecode = strencode.decode()
        para_data['text']=content
        data.append(para_data)
    
    return data

def mask_image(df,page_image):
    for x in df.iloc[:,:4].astype(int).values:
        page_image = cv2.rectangle(page_image, tuple(x[:2]),tuple(x[2:]),[255,255,255],-1)
    return page_image

def clean_image(image, ocr_df, page_index):
    page_id=page_index
    ocr_df=is_bulit(ocr_df)
    ocr_df=is_header_footer(ocr_df)
    select_page = ocr_df[ocr_df.page_index==page_id]
    remove_header_footer = select_page[select_page.is_header_footer==True]
    remove_bulit = select_page[select_page.is_start_bulit==True]
    
    image_header=mask_image(remove_header_footer, image)
    _image = mask_image(remove_bulit, image_header)
    
    return _image


def correct_df_area(page_ocr,page_index,page_area):
    page_ocr=page_ocr[page_ocr.page_index==page_index]
    sent,bold=extract_line(page_ocr,page_area)
    page_area["data"]=sent
    page_area["is_bold"]=[all(list_bold) for list_bold in bold]
    page_area["clean_data"]=[" ".join(x) for x in sent]
    page_area['word_length']=page_area.data.apply(lambda x: len(x))
    page_area['is_upper']=page_area.clean_data.apply(lambda x: x.isupper())
    update_df=check_title(page_area)
    update_df["is_duplicate"]=update_df.clean_data.duplicated()
    
    return update_df[update_df.is_duplicate==False]


def load_ocr(filename=None):
    pdf_layout, pdf_images = sop.load_pdf(filename,load_images=True)
    ocr_data=[]
    for page_index,ocr in enumerate(pdf_layout):
        image=np.array(pdf_images[page_index])
        df=ocr.to_dataframe()
        df["page_index"]=page_index
        df["height"] = abs(df.y_1-df.y_2)
        df['width'] = abs(df.x_1-df.x_2)
        df['page_width'] = image.shape[1]
        df['page_height'] = image.shape[0]
        df['clen_type']=df.type.apply(lambda x: x.lower())
        df['is_bold']=df.clen_type.str.contains('bold')
        ocr_data.append(df)
    ocr_df=pd.concat([x for x in ocr_data])
    return ocr_df,pdf_images
    

def layout_model(image,page_index=0, table=True):
    if table:
        table_layout=model_table.detect(image)
        image=mask_image(table_layout.to_dataframe(),image)
        table_layout= table_layout.to_dataframe()
    layout=model_TT.detect(image)
    df=layout.to_dataframe()
    df = pd.concat([df,table_layout])
#     df["is_overlap"]=df.index.isin(NMS(df.iloc[:,:4].values))
#     df=df[df.is_overlap==True] #rm for subtitle
    df["page_index"] = page_index
    df["height"] = abs(df.y_1-df.y_2)
    df['width'] = abs(df.x_1-df.x_2)
    df['page_width'] = image.shape[1]
    df['page_height'] = image.shape[0]
    df = df.sort_values(by=['y_1'], ascending=True)
    
    return df


