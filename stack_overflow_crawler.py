import urllib.request
from bs4 import BeautifulSoup
import re
import csv


url_stem = "https://movies.stackexchange.com/questions?tab=votes&pagesize=50"
num_pages = 1


def readStackExchangeURL(url):
    response = urllib.request.urlopen(url)
    page = response.read()
    soup = BeautifulSoup(page)
    return soup


def findQuestionLinks(soup):
    links = []
    for link in soup.findAll('a', attrs={'href': re.compile("^/questions/\d+")}):
        links.append(link.get('href'))
    return links


def findQuestion(soup):
    return soup.find("meta", property="og:title")["content"]


# clean_answer[0] is the question description
# other elements in clean_answer are true answers to the question
def findAnswers(soup):
    answers = soup.findAll("div", {"class": "post-text"})
    clean_answers = []
    for answer in answers:
        answer = answer.text.strip()
        clean_answers.append(answer.replace('\n', ''))
    return clean_answers


def readQAs():
    data = []
    for i in range(1, num_pages + 1):
        question_list_url = url_stem + "&page=" + str(i)
        question_list_soup = readStackExchangeURL(question_list_url)
        question_links = findQuestionLinks(question_list_soup)
        for link in question_links:
            soup_one_question = readStackExchangeURL("https://movies.stackexchange.com/" + link)
            data_one_question = []
            data_one_question.append(findQuestion(soup_one_question))
            data_one_question.extend(findAnswers(soup_one_question))
            data.append(data_one_question)
    return data


with open('output.csv', 'w+', newline='', encoding='utf-8') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(data)
