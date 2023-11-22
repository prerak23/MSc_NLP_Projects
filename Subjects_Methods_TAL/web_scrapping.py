import urllib
from bs4 import BeautifulSoup
import collections

def html_recovery(url):
	'''
	purpose : recovery of html content from the website

	param : url (string) which contains the url of the website

	return : the html of the website
	'''
	req = urllib.request.Request(url,headers={'User-Agent' : 'Mozilla/5.0'})
	html_doc = urllib.request.urlopen(req).read()
	return(BeautifulSoup(html_doc,'html.parser'))


def web_tag_a(url,class_title,tag):
	'''
	purpose: create a list containing the titles of the different newspaper sites contained in a h3-a

	param : url (string) which contains the url of the website
			class_title which contains the html class of titles
			taag which contains the tag containing titles

	return : list_title
	'''

	#recovery of html content from the site
	page = html_recovery(url)

	'''recovery of h3 titles of the class passed in parameter
	for one website, the only change is the use of h2 instead of h3
	'''
	list1 = page.findAll(tag,class_title)
	list_title = []

	'''depending of the website, title is contained in a span or not
		that's why we test it with the if and we catch the "AttributeError"
	'''
	for item in list1:
		try:
			if(tag[0]=='h'):
				if(item.a.span):
					list_title.append(item.a.span.find_next(string=True).find_next(string=True))
				elif(item.a):
					list_title.append(item.a.find_next(string=True))	
				else:
					print('nothing')
			elif(url=='https://www.independent.co.uk/?CMP=ILC-refresh'):
				list_title.append(item.find_next(string=True)[23:-8])
			elif(url=='https://www.mercurynews.com'):
				list_title.append(item.find_next(string=True)[4:-2])


		except AttributeError as e:
			print('misformatted title')

	return(list_title)



if __name__ == '__main__':

	#recovery titles from newspapers' website with the same formatting (tag-a-eventually span)
	list_titre = web_tag_a('https://www.telegraph.co.uk','list-of-entities__item-body-headline','h3')
	list_titre += web_tag_a('https://www.thetimes.co.uk','Item-headline Headline--m','h3')
	list_titre += web_tag_a('https://wsj.com','wsj-headline dj-sg wsj-card-feature heading-3','h3')
	list_titre += web_tag_a('https://www.dailymail.co.uk/home/index.html','linkro-darkred','h2')
	list_titre += web_tag_a('https://www.independent.co.uk/?CMP=ILC-refresh', 'headline', 'div')
	list_titre += web_tag_a('https://www.mercurynews.com', 'dfm-title', 'span')

	print(list_titre)

	f = open('news_title.txt', 'w')
	for elem in list_titre:
		f.write(elem + '\n')
	f.close()
