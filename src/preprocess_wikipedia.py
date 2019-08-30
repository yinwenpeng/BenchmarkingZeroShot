from wikipedia import WikipediaPage
import json
import codecs

def build_wiki_category_dataset():
    readfile = codecs.open('/export/home/Dataset/wikipedia/parsed_output/tokenized_wiki/tokenized_wiki2categories.txt', 'r', 'utf-8')
    # writefile = codecs.open('/export/home/Dataset/wikipedia/parsed_output/tokenized_wiki/tokenized_wiki2categories.txt', 'w', 'utf-8')
    co = 0
    for line in readfile:
        try:
            line_dic = json.loads(line)
        except ValueError:
            continue
        print(line_dic.get('categories'))
        print(line_dic.get('text'))
        break
    #     title_id = line_dic.get('id')
    #     article = WikipediaPage(pageid=title_id)
    #     type_list = article.categories
    #     line_dic['categories'] = type_list
    #     writefile.write(json.dumps(line_dic)+'\n')
    #     co+=1
    #     if co == 10:
    #         break
    # writefile.close()
    # readfile.close()


if __name__ == '__main__':
    build_wiki_category_dataset()
