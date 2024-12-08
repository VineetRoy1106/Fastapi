# from requests_html import HTMLSession

# class Scraper():

#     def scrape_data(self,tag):
#         url = f"https://quotes.toscrape.com/tag/{tag}"
#         s= HTMLSession()
#         r = s.get(url)
#         print(r.status_code)

#         qlist = []

#         quotes = r.html.find('div.quote')

#         for q in quotes:
#             items={
#                 "text": q.find('span.text', first = True).text.strip(),
#                 "Author":q.find("small.author",first =True).text.strip()
#             }

#             print(items)
#             qlist.append(items)
# # quotes=  Scraper()
# # quotes.scrape_data("humor")




from requests_html import HTMLSession

class Scraper:
    def scrape_data(self, url:str):
        # Initialize session
        session = HTMLSession()
        # url = 'https://quotes.toscrape.com/tag/truth/'
        r = session.get(url)
        
        # Check the status code
        if r.status_code != 200:
            return {"error": "Failed to fetch the content from the URL"}

        # Scrape the required content
        quotes = r.html.find('div.quote')
        scraped_data = []

        for q in quotes:
            items = {
                "text": q.find('span.text', first=True).text.strip(),
                "author": q.find("small.author", first=True).text.strip()
            }
            print(items)
            scraped_data.append(items)

        return scraped_data

quotes=  Scraper()
quotes.scrape_data("https://quotes.toscrape.com/tag/humor/")