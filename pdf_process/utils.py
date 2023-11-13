## please refer to requirements.txt
## package names that should be installed are different from the names imported below.
import fitz, os, shutil
import pandas as pd
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import HTMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from html.parser import HTMLParser


def from_pdf_to_images(pdf_filename, folder_to_save, include_cls_image = False, initialize_folder = True):
    """For a given pdf file, we extract all of images of this file and save them as png format.

    Args:
        pdf_filename (str): name of pdf file to extract.
        folder_to_save (str): name of folder to save image files.
        include_cls_image (bool, optional): 
            Each image has common image at top of the pdf file. 
            I think this is a character image of this patent institute.
            In my opinion, there is a possibility that it can be used as a [CLS] token for each image, depending on the machine learning model we use.
            Defaults to False.
        initialize_folder (bool, optional):
            If True, it removes all of contents and folder('folder_to_save')
            If False, it raises error if there is already exists folder or file we got as argument('pdf_filename' or 'folder_to_save').
    """
    folder_to_save = str(folder_to_save)
    if initialize_folder:
        if os.path.exists(folder_to_save):
            shutil.rmtree(folder_to_save)
        os.makedirs(folder_to_save)
    ignored_CLS = False if include_cls_image == True else True
    
    doc = fitz.open(pdf_filename)
    for i in range(len(doc)):
        for img in doc.get_page_images(i):
            if ignored_CLS == False:
                ignored_CLS = True
                continue
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if xref > 100:
                continue
            if pix.n < 5:      
                pix.save(os.path.join(folder_to_save, "p%s-%s.png") % (i, xref))
            else:        
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                pix1.save(os.path.join(folder_to_save, "p%s-%s.png") % (i, xref))
                pix1 = None
            pix = None
            


def from_pdf_to_html(pdf_path, folder_to_save):
    
    if not os.path.exists(os.path.join(os.getcwd(), 'html')):
        os.makedirs(os.path.join(os.getcwd(), 'html'), exist_ok=True)
    rsrcmgr = PDFResourceManager() 
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    
    f = open(folder_to_save, 'wb')
    device = HTMLConverter(rsrcmgr, f, codec=codec, laparams=laparams)
    
    fp = open(pdf_path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0 
    caching = True
    pagenos=set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)
    fp.close()
    device.close()
    str = retstr.getvalue()
    retstr.close()
    f.close()
    
def from_html_to_df(file_path, name):

    with open(file_path) as f:
        parser = patent_HTMLParser(name)
        parser.feed(f.read())
    
    # for key, val in parser.contents.items():
    #     print(f"key: {key}")
    #     print(f"val: {val}")
    #     print()
    
    df = pd.DataFrame.from_dict([parser.contents])
    # print("df:")
    # print("columns: ",df.columns)
    # print(df.iloc[0, :])
    
    return df
    # if csvfile_path in os.listdir():
    #     # print("cvsfile exist")
    #     prev_df = pd.read_csv(csvfile_path, encoding = 'utf-8')
    #     # print("prev_df:")
    #     # print(prev_df)
    #     # print()
    #     df = pd.concat([prev_df, df], ignore_index=True)
    
    # df.to_csv(csvfile_path, index=False, encoding='utf-8')
    
class patent_HTMLParser(HTMLParser):
    def __init__(self, name):
        HTMLParser.__init__(self)
        self.contents = {'id':name,
                         '요약':'',
                         '대표도':'',
                         '청구범위':'',
                         '기술분야':'',
                         '배경기술':'',
                         '해결하려는과제':'',
                         '과제의해결수단':'',
                         '발명의효과':'',
                         '도면의간단한설명':'',
                         '발명을실시하기위한구체적인내용':'',
                         '부호의설명':'',}
        self.current_title_count = 0
        self.curr_is_content = False
        self.curr_is_title = False
        self.curr_title = None
        self.count = 0
        self.forbid_after = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'br':
            return
        self.count +=1
        self.curr_is_title = self.is_title_determine_by_starttag(tag, attrs)
        self.curr_is_content = self.is_content_determine_by_starttag(tag, attrs)
        # print(f"tag:{tag}")
        # print(f"attrs:{attrs}")
        

    def handle_data(self, data):
        self.count +=1
        # if self.count > 200:
        #     raise f"self.count = {self.count}"
        # print(data)
        # print()
        if self.curr_is_title:
            for key in self.contents.keys():
                if is_substring(key, data.replace(' ', '')) and len(key)*4 > len(data.replace(' ', '')):
                    self.curr_title = key
                    # print(f"self.curr_title: {self.curr_title}")
            else:
                self.curr_is_title = False
        
        if self.curr_is_content:
            is_content = self.is_content_determine_by_data(data)
            if is_content and self.curr_title != None and not is_substring(self.curr_title, data.replace(' ', '')):
                if self.curr_title != '요약':
                    self.contents[self.curr_title] += data.replace('\n', '')
                else:
                    self.contents[self.curr_title] += data.replace('\n', '')
                    
    def is_content_determine_by_starttag(self, tag, attrs):
        """
        본문이 되기 위한 필요조건.
        1. span tag이어야 한다.(요약 항목 빼고)
        """
        is_content = True
        if self.curr_title != '요약':
            if tag != 'span' and self.curr_title != '요약':
                is_content = False
            if len(attrs) == 0:
                return False
            else:
                if len(attrs[0])==1:
                    return False
            if 'BatangChe' not in attrs[0][1] and self.curr_title != '요약':
                is_content = False

        # else:
        #     if tag 
        
        return is_content

        
    def is_title_determine_by_starttag(self, tag, attrs):
        """
        title이 되기 위한 필요조건.
        1. span tag이어야 한다.
        2. font_size:9px이어야 한다.
        """
        is_title = True
        if tag != 'span':
            is_title = False
        if len(attrs) == 0:
            return False
        else:
            if len(attrs[0])==1:
                return False
        if '9px' not in attrs[0][1] or 'BatangChe' not in attrs[0][1]:
            is_title = False
        return is_title

    def is_content_determine_by_data(self, data):
        is_content = True
        if data.startswith('[') and data.endswith(']'):
            is_content = False
        if data.startswith('공개특허'):
            is_content = False
        if data.startswith('-') and data.endswith('-'):
            is_content = False
        if '뒷면에 계속' in data:
            is_content = False
        
        for key, val in {'특허권자':3, '발명자':7, '대리인':3, '심사관':0}.items():
            if key in data and self.curr_title == '요약':
                is_content = False
                self.forbid_after = val
        
        if self.forbid_after != 0:
            self.forbid_after -=1
            is_content = False
                
        return is_content
    
def is_substring(a, b):
    if a in b:
        return True
    else:
        False