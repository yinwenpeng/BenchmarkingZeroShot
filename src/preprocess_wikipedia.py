from wikipedia import WikipediaPage
import json
import codecs

def build_wiki_category_dataset():
    readfile = codecs.open('/export/home/Dataset/wikipedia/parsed_output/tokenized_wiki/tokenized_wiki.txt', 'r', 'utf-8')
    writefile = codecs.open('/export/home/Dataset/wikipedia/parsed_output/tokenized_wiki/tokenized_wiki2categories.txt', 'w', 'utf-8')
    co = 0
    for line in readfile:
        try:
            line_dic = json.loads(line)
        except ValueError:
            continue

        try:
            # title = line_dic.get('title')
            title_id = line_dic.get('id')
            article = WikipediaPage(pageid=title_id)
        except AttributeError:
            continue
        type_list = article.categories
        # print(type_list)
        line_dic['categories'] = type_list
        writefile.write(json.dumps(line_dic)+'\n')
        co+=1
        if co % 5 == 0:
            print(co)
        if co == 100000:
            break
    writefile.close()
    readfile.close()
    print('over')


if __name__ == '__main__':
    build_wiki_category_dataset()
