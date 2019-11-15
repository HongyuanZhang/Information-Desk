import string
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import requests
import json
import json_lines
import numpy as np
import time

import torch
from torch.utils.data import Dataset, DataLoader

engine_id=''
api_key=''

google_nq=list()

charset=list(string.printable)
charset.append('SOS') # start of string token
charset.append('EOS') # end of string token
int2char=dict(enumerate(charset)) # dict from int to char
char2int={ch:ii for ii,ch in int2char.items()} # dict from char to int

def clean_text(s):
    '''
    remove non-printable chars
    '''
    s=''.join(filter(lambda x: x in charset, s))
    return s

def clean_html(page_raw):
    soup = BeautifulSoup(page_raw,features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    if soup.body is not None:
        text = soup.body.get_text(separator=' ')
    else:
        text = soup.get_text(separator=' ')
    lines = (line.strip() for line in text.splitlines())
    #chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
    #text = '\n'.join(chunk for chunk in chunks if chunk)
    text = '\n'.join(line for line in lines if line)
    text=clean_text(text)
    return text

def get_html(url):
    try:
        page = urllib.request.urlopen(url)
    except ValueError:
        return None
    except urllib.error.URLError:
        return None
    except ConnectionResetError:
        return None
    except TimeoutError:
        return None

    # raw contents read from the page
    page_raw=page.read()
    text=clean_html(page_raw)
    return text

def str2int(s):
    '''
    turn a string to int array
    '''
    s=clean_text(s)
    arr=[char2int[ss] for ss in s]
    return arr

def int2str(arr):
    s=str(''.join([int2char[i] for i in arr]))
    return s

def get_webpages(q):
    '''
    given a question str
    get relative web page text
    '''
    q=urllib.parse.quote_plus(q)
    query="https://www.googleapis.com/customsearch/v1?key="+api_key+"&cx="+engine_id+"&q="+q
    req = requests.get(query)
    result = req.text
    data = json.loads(result)
    results=data['items']
    links=[r['link'] for r in results]
    texts=[str(get_html(l)) for l in links]
    return texts

def one_hot(arr):
    '''
    convert an int arr to one hot np arr
    '''
    onehot=np.zeros((len(arr),len(charset)),dtype=np.uint8)
    for i in range(len(arr)):
        onehot[i,arr[i]]=1
    return onehot

class GoogleNQ(Dataset):
    def __init__(self,path):
        starttime=time.time()
        print("loading google nq...")
        count=0
        with open(path, 'rb') as f:
            for item in json_lines.reader(f):
                '''
                    '''
                if count>=100:
                    break
                nq=dict()
                nq['q']=clean_text(item['question_text'])
                document=item['document_text']
                if item['annotations'][0]['yes_no_answer']=='NONE':
                    start=int(item['annotations'][0]['long_answer']['start_token'])
                    if start<0:
                        continue
                    end=int(item['annotations'][0]['long_answer']['end_token'])
                    answer=" ".join(document.split(" ")[start:end])
                    nq['a']=clean_html(answer)
                else:
                    nq['a']=item['annotations'][0]['yes_no_answer']
                nq['d']=clean_html(document)
                google_nq.append(nq)
                count+=1
        print("finished loading!")
        endtime=time.time()
        print(endtime-starttime)

    def __len__(self):
        return len(google_nq)

    def head_tail(self,s):
        ss=[char2int['SOS']]
        ss.extend(s)
        ss.append(char2int['EOS'])
        return ss

    def __getitem__(self,idx):
        item = google_nq[idx]
        q=str2int(item['q'])
        d=str2int(item['d'])
        a=str2int(item['a'])
        q=self.head_tail(q)
        d=self.head_tail(d)
        a=self.head_tail(a)
        qd=q
        qd.extend(d)
        qd=torch.LongTensor(qd)
        a=torch.LongTensor(a)
        return (qd,a)

def test_google():
    dataset=GoogleNQ('simplified-nq-train.jsonl')
    dataloader=DataLoader(dataset,batch_size=1,shuffle=True,num_workers=1)
    count=0
    for _,(qd,a) in enumerate(dataloader):
        print(qd.shape)
        print(a.shape)
        print(a)
        print()
        count+=1
        if count>5:
            break

if __name__ == '__main__':
    '''
    texts=get_webpages("why is the sky blue?")
    print(len(texts))
    print(texts[0])
    '''
    '''
    arr=str2int("why is the sky blue?")
    s=int2str(arr)
    print(arr)
    print(s)
    '''
    '''
    count=0
    with open('simplified-nq-train.jsonl', 'rb') as f:
        for item in json_lines.reader(f):
            if count>5:
                break
            document=item['document_text']
            print(item['question_text'])
            start=int(item['annotations'][0]['long_answer']['start_token'])
            end=int(item['annotations'][0]['long_answer']['end_token'])
            print(item['annotations'][0]['long_answer']['start_token'])
            print(item['annotations'][0]['long_answer']['end_token'])
            answer=" ".join(document.split(" ")[start:end])
            print(clean_html(answer))
            print(item) 
            count+=1
            '''
    test_google()
