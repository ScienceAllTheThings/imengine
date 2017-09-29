from bs4 import BeautifulSoup as Soup
import urllib2
import json
import urllib

def get_links(query_string, num_images):
    links = []
    for i in range(0,num_images,100):
        url = 'https://www.google.com/search?ei=1m7NWePfFYaGmQG51q7IBg&hl=en&q='+query_string+'&tbm=isch&ved=0ahUKEwjjovnD7sjWAhUGQyYKHTmrC2kQuT0I7gEoAQ&start='+str(i)+'&yv=2&vet=10ahUKEwjjovnD7sjWAhUGQyYKHTmrC2kQuT0I7gEoAQ.1m7NWePfFYaGmQG51q7IBg.i&ijn=1&asearch=ichunk&async=_id:rg_s,_pms:s'
        request = urllib2.Request(url, None, {'User-Agent': 'Mozilla/5.0'})
        json_string = urllib2.urlopen(request).read()
        page = json.loads(json_string)
        html = page[1][1]
        new_soup = Soup(html,'html')
        imgs = new_soup.find_all('img')
        for j in range(len(imgs)):
            links.append(imgs[j]["src"])
    return links

def get_images(links,pre):
    for i in range(len(links)):
        urllib.urlretrieve(links[i], "./images/"+str(pre)+str(i)+".jpg")

terms = ["cars","numbers","scenery","people","dogs","cats","animals"]
for x in range(len(terms)):
    all_links = get_links('animated+'+terms[x],1000)
    get_images(all_links,x)





