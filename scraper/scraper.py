import requests
from bs4 import BeautifulSoup
import shelve
from collections import deque, defaultdict
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin
import urllib3


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


class Scraper:    

    def __init__(self) -> None:             
        self.curr_url = ""
        self.db_name = "links_queue"        
        self.db = shelve.open('queue_data.db', writeback=True)              
        # global_rules = {"http": {key: {}}, "https": {key: {}}}
        # self.global_rules = {"http": defaultdict(dict), "https": defaultdict(dict)}        
        self.global_rules = {
            "http": defaultdict(lambda: {"allowed": [], "disallowed": []}),
            "https": defaultdict(lambda: {"allowed": [], "disallowed": []})
        }
        # self.global_rules = {
        #     "http": defaultdict(defaultdict(list)),
        #     "https": defaultdict(defaultdict(list))
        # }
        # self.robot_parser = RobotFileParser()

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
        

    def explore_paths(self):
        ...
    
    def extract_links(self):
        ...

    def store_links(self):
        if self.db_name not in self.db:
            print(f"Creating a SetQueue for key: {self.db_name}!")
            self.db[self.db_name] = SetDeque()

        # self.db[self.db_name].append("data1")
        # self.db.sync()

        # item = self.db['links_queue'].popleft()
        # if item:
        #     print("Dequeud_item:", item)

    def crawl_website(self, limit = 100):
        self.db[self.db_name].append("https://www.quora.com/")
        # self.db[self.db_name].append("https://www.google.com/")
        self.curr_url = self.db[self.db_name].popleft()
        # Fetch robots.txt and explore all those paths
        self.check_robots_txt()

        # Store the links found on the pages to our db
        # Pop a link and start exploring again



sc = Scraper()
sc.crawl_website()
# r = requests.get('')

# soup = BeautifulSoup()