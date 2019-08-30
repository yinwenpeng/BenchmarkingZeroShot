from wikipedia import WikipediaPage



article = WikipediaPage(pageid='39526180')
print(article.categories)
sections = article.sections
print('sections titles:', sections)
for sec in sections:
    print(sec, '-->', article.section(sec))
