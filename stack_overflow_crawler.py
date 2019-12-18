import urllib.request
from bs4 import BeautifulSoup
import re
import requests

# stem url, the page of list of Stack Exchange questions
url_stem = "https://movies.stackexchange.com/questions?tab=votes&pagesize=50"
# number of pages of list of questions to crawl
num_pages = 100
data = []


def readStackExchangeURL(url):
    '''
    read url and return its soup
    '''
    response = urllib.request.urlopen(url)
    page = response.read()
    soup = BeautifulSoup(page)
    return soup


def findQuestionLinks(soup):
    '''
    find links of questions in soup
    '''
    links = []
    for link in soup.findAll('a', attrs={'href': re.compile("^/questions/\d+")}):
        links.append(link.get('href'))
    return links


def findQuestion(soup):
    '''
    find question title in a specific question page
    '''
    return soup.find("meta", property="og:title")["content"]


# clean_answer[0] is the question description
# other elements in clean_answer are true answers to the question
def findAnswers(soup):
    '''
    find people's answers in a specific question page
    '''
    answers = soup.findAll("div", {"class": "post-text"})
    clean_answers = []
    # compile cleaned answers
    for answer in answers:
        answer = answer.text.strip()
        clean_answers.append(answer.replace('\n', ''))
    return clean_answers


def readQAs():
    '''
    read up to num_pages lists of questions and their answers
    '''
    for i in range(1, num_pages + 1):
        # the i-th list of questions
        question_list_url = url_stem + "&page=" + str(i)
        question_list_soup = readStackExchangeURL(question_list_url)
        question_links = findQuestionLinks(question_list_soup)
        # process each question link
        for link in question_links:
            soup_one_question = readStackExchangeURL("https://movies.stackexchange.com/" + link)
            data_one_question = []
            data_one_question.append(findQuestion(soup_one_question))
            data_one_question.extend(findAnswers(soup_one_question))
            # data: [[Q1, A11, A12, ...],
            #        [Q2, A21, A22, ...],
            #        ...                ]
            data.append(data_one_question)


def compile_fake_answers(questions):
    '''
    get answer candidates corresponding to questions
    '''
    fake_answers = []
    for q in questions:
        # search using google
        url = "https://www.google.dz/search?q="+q.replace(' ', '+')
        # read search result page
        page = requests.get(url)
        soup = BeautifulSoup(page.content, parser='html.parser', features='lxml')
        # get search result links
        links = soup.find_all("a", href=re.compile("(?<=/url\?q=)(htt.*://.*)"))
        # get top search result
        top_result_link = re.split(":(?=http)", links[0]["href"].replace("/url?q=", ""))[0]
        top_result = requests.get(top_result_link).content
        soup = BeautifulSoup(top_result, parser='html.parser', features='lxml')
        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()  # rip it out
        # get text
        text = soup.get_text()
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = ' '.join(i for i in chunks if re.match('^[A-Z][^?!.]*[?.!]$', i) is not None)
        fake_answers.append(text)
    return fake_answers
