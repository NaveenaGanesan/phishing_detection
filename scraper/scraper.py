from shutil import which
import shelve, scrapy
import requests
from bs4 import BeautifulSoup
from collections import deque, defaultdict
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin
import urllib3
from scrapy_selenium import SeleniumRequest
from scrapy.crawler import CrawlerProcess

class SetDeque:
    def __init__(self) -> None:
        self.s = set()
        self.d = deque()

    def append(self, item: str) -> None:
        if item not in self.s:
            self.d.append(item)
            self.s.add(item)

    def popleft(self) -> str:
        item = None
        if self.d:
            item = self.d.popleft()
            self.s.remove(item)
        return item


class MySpider(scrapy.Spider):
    def __init__(self, scraper, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.scraper = scraper

    def start_request(self, url):
        yield SeleniumRequest(
            url=url,
            callable = self.parse_request,
            wait_time=10
        )

    def parse_request(self, response):
        driver = response.meta['driver']
        links = driver.find_elements_by_xpath("//a[@href]")

        for link in links:
            url = link.get_attribute("href")
            if self.scraper.is_scraping_allowed(url):
                self.scraper.db[self.scraper.db_name].append(url)

        response.release_driver()

class Scraper:

    def __init__(self) -> None:
        self.curr_url = ""
        self.db_name = "links_queue"
        self.db = shelve.open('queue_data.db', writeback=True)
        if self.db_name not in self.db:
            print(f"Creating a SetQueue for key: {self.db_name}!")
            self.db[self.db_name] = SetDeque()

        # global_rules = {"http": {key: {}}, "https": {key: {}}}
        # self.global_rules = {"http": defaultdict(dict), "https": defaultdict(dict)}
        self.global_rules = {
            "http": defaultdict(lambda: {"allowed": [], "disallowed": []}),
            "https": defaultdict(lambda: {"allowed": [], "disallowed": []})
        }

        self.links = []

        self.settings = {
            "SELENIUM_DRIVER_NAME": 'chrome',
            "SELENIUM_DRIVER_EXECUTABLE_PATH": which('chromedriver'),
            "SELENIUM_DRIVER_ARGUMENTS" : ['--headless'],
            "DOWNLOAD_MIDDLEWARES": {
                'scrapy_selenium.SeleniumMiddleware': 800
            },
            "DOWNLOAD_DELAY": 2,  # Delay between requests (in seconds)
            "CONCURRENT_REQUESTS": 16,  # Number of concurrent requests
            "AUTOTHROTTLE_ENABLED": True,  # Enable AutoThrottle
            "AUTOTHROTTLE_START_DELAY": 5,  # Initial download delay
            "AUTOTHROTTLE_MAX_DELAY": 60,  # Maximum download delay
            "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,  # Average number of requests to send in parallel
        }


    def __del__(self):
        try: self.db.close()
        except: pass

    def _parse_url_(self, url=""):
        parsed_url = urlparse(url if url != "" else self.curr_url)
        return parsed_url

    def check_robots_txt(self):
        data = ""
        parsed_url = self._parse_url_(self.curr_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        domain_robot_url = urljoin(base_url, "/robots.txt")

        def fetch_robots_file():
            nonlocal data, domain_robot_url
            try:
                http = urllib3.PoolManager(cert_reqs='CERT_NONE')
                response = http.request("GET", domain_robot_url)
                if response.status == 200:
                    data = response.data.decode("utf-8")
                else: print("Could not fetch Robots.txt:", response.status_code)
            except Exception as e: print("Error fetching robots:", e)

        def process_robots_txt():
            nonlocal data
            data = data.splitlines()
            rules = defaultdict(list)
            user_agent = None
            for line in data:
                line = line.strip().lower()
                if line.startswith('user-agent:'):
                    user_agent = line.split(":")[1].strip()
                elif user_agent == "*":
                    if line.startswith("disallow:"): rules["disallowed"].append(line.split(":")[1].strip())
                    elif line.startswith("allow:"): rules["allowed"].append(line.split(":")[1].strip())

            return rules


        fetch_robots_file()
        rules = process_robots_txt()
        scheme, netloc = parsed_url.scheme, parsed_url.netloc
        self.global_rules[scheme][netloc]["allowed"].extend(rules["allowed"])
        self.global_rules[scheme][netloc]["disallowed"].extend(rules["disallowed"])
        print(self.global_rules[scheme][netloc])

    def is_scraping_allowed(self, url):
        parsed_url = self._parse_url_(url)
        scheme, netloc, path = parsed_url.scheme, parsed_url.netloc, parsed_url.path
        # Contains boh allowed and dissallowed rules for "https", "quora.com"
        rules = self.global_rules[scheme][netloc]
        if not isinstance(rules["disallowed"], set): rules["disallowed"] = set(rules["disallowed"])
        if not isinstance(rules["allowed"], set): rules["allowed"] = set(rules["allowed"])
        if any(path.startswith(disallowed) for disallowed in rules["disallowed"]): return False
        return True

    def crawl_website(self, limit = 10):
        # Adding sample website to start scraping with
        self.db[self.db_name].append("https://www.quora.com/")

        count = 0
        while count < limit:
            if self.db[self.db_name]:
                self.curr_url = self.db[self.db_name].popleft()
                print(f"Crawling: {self.curr_url}")

                # Fetch robots.txt and explore all those paths
                self.check_robots_txt()

                # Store the links found on the pages to our db
                yield from self.start_request(self.curr_url)

                count += 1
            else:
                print("No links to pop")
                break

    def run_spider(self):
        process = CrawlerProcess(self.settings)
        process.crawl(self.crawl_website)
        process.start()

if __name__ == "__main__":
    sc = Scraper()
    sc.run_spider()