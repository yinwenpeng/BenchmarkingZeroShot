from wikipedia import WikipediaPage
import json
import codecs

def build_wiki_category_dataset():
    readfile = codecs.open('/export/home/Dataset/wikipedia/parsed_output/tokenized_wiki/tokenized_wiki.txt', 'r', 'utf-8')
    writefile = codecs.open('/export/home/Dataset/wikipedia/parsed_output/tokenized_wiki/tokenized_wiki2categories.txt', 'w', 'utf-8')
    for line in f:
        try:
            line_dic = json.loads(line)
        except ValueError:
            continue
        # title = line_dic.get('title')
        title_id = title2id.get('id')
        article = WikipediaPage(pageid=title_id)
        type_list = article.categories
        print(type_list)
        exit(0)


if __name__ == '__main__':
    build_wiki_category_dataset()
