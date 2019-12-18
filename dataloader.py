import string
import urllib.error
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import requests
import json
import json_lines
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# for google custom search api
# if blank, fill in your own
engine_id=' '
api_key=' '

# key sentence extractor is a char-based rnn model
charset=list(string.printable) # characters that we consider
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
    '''
    clean up web pages, remove tags and clean up the text
    '''
    soup = BeautifulSoup(page_raw, features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    if soup.body is not None:
        lines=list(soup.body.stripped_strings)
    else:
        lines=list(soup.stripped_strings)
    text = ' '.join(line for line in lines)
    text=clean_text(text)
    return text


def get_html(url):
    '''
    download the web page given url and return the cleaned page content
    '''
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
    # clean page contents
    text=clean_html(page_raw)
    return text


def str2int(s):
    '''
    turn a string to int array
    '''
    s=clean_text(s) # remove uncommon characters
    arr=[char2int[ss] for ss in s]
    return arr


def int2str(arr):
    '''
    turn an int array into a string
    '''
    s=str(''.join([int2char[i] for i in arr]))
    return s


def get_webpages(q):
    '''
    given a question str
    get a list of related web pages
    '''
    q=urllib.parse.quote_plus(q) # Replace special characters in string using the %xx escape
    query="https://www.googleapis.com/customsearch/v1?key="+api_key+"&cx="+engine_id+"&q="+q
    req = requests.get(query) # search using google
    # get search result links
    result = req.text
    data = json.loads(result)
    results=data['items']
    links=[r['link'] for r in results]
    texts=[str(get_html(l)) for l in links] # get text content in these links
    return texts


def one_hot(arr):
    '''
    convert an int arr to one hot numpy arr
    '''
    onehot=np.zeros((len(arr),len(charset)),dtype=np.uint8)
    for i in range(len(arr)):
        onehot[i,arr[i]]=1
    return onehot


# bucket sizes for document length
buc_lim=[5000,10000,15000,20000,30000,40000,50000,60000]
# limit number of characters for questions
q_lim=100
# limit number of characters for answers
a_lim=600


def get_bucs(path, buc_lim, q_lim, a_lim, max_num):
    '''
    path: json data file path
    buc_lim, q_lim, a_lim: see above comments
    max_num: max number of entries read

    pre-process the Google NQ data set
    divide the entries into buckets according to document length
    '''
    buc=[list() for i in range(len(buc_lim))] # a list of empty lists
    print("loading google nq...")
    count=0 # count for entries read
    # open the google nq data set file
    with open(path, 'rb') as f:
        for item in json_lines.reader(f):
            if count>=max_num:
                break
            nq=dict()
            nq['q']=clean_text(item['question_text'])
            # filter out overly long questions
            if len(nq['q'])>q_lim:
                continue
            document=item['document_text']
            # if not a yes-or-no question, store answer's word start/end indices
            if item['annotations'][0]['yes_no_answer']=='NONE':
                start=int(item['annotations'][0]['long_answer']['start_token'])
                if start<0:
                    continue
                end=int(item['annotations'][0]['long_answer']['end_token'])
                # compile corresponding answer
                doc_tokens=document.split(" ")
                dt_len=len(doc_tokens)
                answer=" ".join(doc_tokens[start:end])
                nq['a']=clean_text(answer)
            else:
                continue
            # filter out overly long answers
            if len(nq['a'])>a_lim:
                continue
            # store cleaned document
            nq['d']=clean_text(document)
            # normalize the indices
            nq['s']=start/dt_len
            nq['e']=end/dt_len
            for i in range(len(buc_lim)):
                # if the document is within length buc_lim[i]
                if len(nq['d'])<=buc_lim[i]:
                    # throw the entry to the corresponding bucket
                    buc[i].append(nq)
                    break
            count+=1
    print("finished loading!")
    # write each bucket into different files
    for i in range(len(buc_lim)):
        with open('buc_test/buc_'+str(buc_lim[i])+'.json', 'w', encoding='utf-8') as f:
            json.dump(buc[i], f, ensure_ascii=True, indent=4)


# Class for the data set google nq
class GoogleNQ(Dataset):
    def __init__(self, buc_lim, q_lim, a_lim, test=False):
        self.buc_lim=buc_lim
        self.q_lim=q_lim
        self.a_lim=a_lim
        if test:
            path='buc_test/buc_'+str(buc_lim)+'.json'
        else:
            path='buc/buc_'+str(buc_lim)+'.json'
        with open(path, 'r') as f:
            self.data = json.load(f) # load data

    def __len__(self):
        return len(self.data)

    # add starting and ending tokens to a string
    def head_tail(self,s):
        ss=[char2int['SOS']]
        ss.extend(s)
        ss.append(char2int['EOS'])
        return ss

    def __getitem__(self, idx):
        '''
        return question, document, question length, document length, start index, end index 
        '''
        item = self.data[idx]
        # convert strings to int arrays
        q=str2int(item['q']) # q: question
        d=str2int(item['d']) # d: document
        s=item['s'] # start index
        e=item['e'] # end index
        q=torch.LongTensor(q)
        d=torch.LongTensor(d)
        s=torch.FloatTensor([s])
        e=torch.FloatTensor([e])
        q_len=torch.LongTensor([q.shape[0]])
        d_len=torch.LongTensor([d.shape[0]])
        # copy q, d into tensors whose length is the specified limit
        d_=d.new(self.buc_lim).long().zero_()
        d_[:d.shape[0]]=d
        q_=q.new(self.q_lim).long().zero_()
        q_[:q.shape[0]]=q
        return (q_, d_, q_len, d_len, s, e)


if __name__ == '__main__':
    # load data into buckets and write buckets into local files
    get_bucs('simplified-nq-train.jsonl', buc_lim, q_lim, a_lim, 20000)
