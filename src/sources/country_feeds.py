COUNTRY_KEYWORDS = {
    "ukraine": ["ukraine","kyiv","kiev","kharkiv","odesa","lviv","zaporizhzhia","dnipro","donetsk","luhansk","crimea"],
    "syria":   ["syria","damascus","aleppo","idlib","deir ez-zor","hasakah","homs","hama","raqqa","latakia"],
    "yemen":   ["yemen","sanaa","aden","hodeidah","taiz","marib","saada","ib","shabwah","hadhramaut"],
}

# Curated + your existing list (trim any duplicates)
GLOBAL_WESTERN_FEEDS = [
    # keep your original entries here (AP, NPR, PBS, CNN, ABC, CBS, NBC,
    # Bloomberg, NYT, WaPo, WSJ, USA Today, LA Times, Politico, RFE/RL, etc.)
    #"https://www.bloomberg.com/authors/AQcJycm0xLU/aliaksandr-kudrytski.rss",
    #"https://www.bloomberg.com/authors/ARw-X6KrIIA/volodymyr-verbianyi.rss",
    #"https://www.bloomberg.com/opinion/authors/AVESaFEtYj8/max-hastings.rss",
    #"https://www.bloomberg.com/opinion/authors/AQtgrnA1PQs/marc-champion.rss",
    #"https://www.bloomberg.com/opinion/authors/AUPRf8lvkug/andreas-kluth.rss",
    #"https://www.bloomberg.com/authors/AXTHuNQ_f8k/mishal-husain.rss",
    #"https://www.bloomberg.com/authors/ASzQCOA5TQg/jennifer-jacobs.rss",
    #"https://www.bloomberg.com/opinion/authors/ATKjVhct25c/hal-brands.rss",
    #"https://news.google.com/rss/search?q=intitle%3AUkraine+site%3Areuters.com&hl=en-US&gl=US&ceid=US%3Aen", # Reuters Ukraine News RSS
    #"https://rss.cnn.com/rss/edition_europe.rss",                     # CNN Europe
    #"https://abcnews.go.com/abcnews/internationalheadlines",          # ABC International
    #"http://online.wsj.com/xml/rss/3_7085.xml",                        # Wall Street Journal
    #"http://rss.csmonitor.com/feeds/usa",                              # Christian Science Monitor - USA
    #"http://feeds.nbcnews.com/feeds/topstories",                       # NBC News Top Stories
    #"https://www.atlanticcouncil.org/issue/conflict/feed/",
    #"http://feeds.nbcnews.com/feeds/worldnews",                        # NBC News World News
    #"http://www.huffingtonpost.com/feeds/verticals/world/index.xml",   # Huffington Post - World News
    #"http://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",        # BBC News - U.S. and Canada
    #"https://news.yahoo.com/rss/us",                                    # Yahoo News - U.S. News
    #"https://rss.news.yahoo.com/rss/world",                             # Yahoo News - World News
    #"http://www.newsweek.com/rss",                                     # Newsweek
    #"http://www.theguardian.com/world/usa/rss",                        # The Guardian - U.S. News
    #"http://www.politico.com/rss/politicopicks.xml",                   # Politico
    #"http://www.newyorker.com/feed/news",                              # New Yorker
    #"http://www.usnews.com/rss/news",                                  # U.S. News
    #"http://www.latimes.com/nation/rss2.0.xml",                        # L.A. Times - National News
    #"https://news.vice.com/rss",                                       # Vice News
    #"https://abcnews.go.com/abcnews/topstories",                       # ABC International
    #"https://rss.nytimes.com/services/xml/rss/nyt/World.xml",          # NYT World
    #"https://feeds.washingtonpost.com/rss/world",                      # Washington Post World
    #"https://feeds.a.dj.com/rss/RSSWorldNews.xml",                     # WSJ World
    #"https://rssfeeds.usatoday.com/usatoday-NewsTopStories",           # USA Today
    #"https://www.latimes.com/world/rss2.0.xml",                        # LA Times World
    #"https://www.politico.com/rss/politics-news.xml"                   # Politico (broad)
    #"https://news.un.org/feed/subscribe/en/news/region/europe/feed/rss.xml", # UN
    #"https://www.wfp.org/news/rss.xml",                                # WFP News RSS
    #"https://www.pbs.org/newshour/feeds/rss",                          # PBS NewsHour
    #"https://feeds.bloomberg.com/europe/news.rss",                     # Bloomberg Europe News
    #"https://feeds.bloomberg.com/europe-politics/news.rss"             # Bloomberg Europe Politics
    "https://www.cbsnews.com/latest/rss/world",                         # CBS World
    "https://www.nbcnews.com/world/rss",                                # NBC World
    "http://time.com/newsfeed/feed/",                                   # Time
    "http://feeds.foxnews.com/foxnews/latest?format=xml",               # Fox News
    "http://www.latimes.com/world/rss2.0.xml",                          # L.A. Times - World News
    "https://rss.nytimes.com/services/xml/rss/nyt/Europe.xml"           # NYT Europe
    "http://rss.cnn.com/rss/edition_world.rss"                          # CNN World
    #"https://www.savethechildren.org/en/press/rss/all"                 # Save the Children News
    #"https://www.savethechildren.org/sitemap-image.xml",                # Save the Children News
    "https://www.unmas.org/en/rss.xml",                                 # United Nations Mine Action Service (UNMAS)
    "https://news.un.org/feed/subscribe/en/news/all/rss.xml",           # UN News
    "https://news.un.org/feed/subscribe/en/news/topic/health/feed/rss.xml",  # UN Health
    "https://news.un.org/feed/subscribe/en/news/topic/human-rights/feed/rss.xml", # UN Human Rights
    "https://news.un.org/feed/subscribe/en/news/topic/humanitarian-aid/feed/rss.xml", # UN Humanitarian Aid
    "https://news.un.org/feed/subscribe/en/news/topic/peace-and-security/feed/rss.xml", # UN News - Peace and Security
    "https://news.un.org/feed/subscribe/en/news/topic/migrants-and-refugees/feed/rss.xml" # UN News - Migrants and Refugees
]

UKRAINE_FEEDS = [
    "https://news.un.org/feed/subscribe/en/news/region/europe/feed/rss.xml", # UN Europe
    "https://feeds.bbci.co.uk/news/topics/c1vw6q14rzqt/rss.xml",             # BBC War in Ukraine
    "https://www.rferl.org/api/zbgvmtl-vomx-tpeq_kmr",                       # RFE/RL News RSS Ukraine War
    "https://www.theguardian.com/world/ukraine/rss",
    "https://www.nytimes.com/svc/collections/v1/publish/https://www.nytimes.com/news-event/ukraine-russia/rss.xml",
    "https://www.atlanticcouncil.org/region/ukraine/feed/"
    "https://www.ft.com/war-in-ukraine?format=rss",
    "https://www.foreignaffairs.com/feeds/tag/War%20in%20Ukraine/rss.xml",
    "https://www.politico.eu/tag/war-in-ukraine/feed/",
    "https://feeds.npr.org/1120755253/rss.xml",                        # NPR
    "https://feeds.npr.org/1082539802/rss.xml",                        # NPR
    "https://feeds.npr.org/1226573564/rss.xml",                        # NPR
    "https://feeds.npr.org/1251330900/rss.xml",                        # NPR
    #"https://theconversation.com/topics/ukraine-invasion-2022-117045/articles.atom",
    #"https://www.ejiltalk.org/category/ukraine/feed/",
    #"https://ukukraine.blogspot.com/feeds/posts/default",
    #"https://uavarta.org/en/category/news-en/feed/",
    #"https://www.airandspaceforces.com/category/russia-ukraine/feed/",
    #"https://euromaidanpress.com/category/russian-aggression/russian-ukrainian-war-news/feed/",
    #"https://defence-blog.com/topics/russia-ukraine/feed",
    #"https://blogs.lse.ac.uk/medialse/category/russia-ukraine-war/feed/",
    #"https://blogs.prio.org/category/ukraine-war/feed/",
]

SYRIA_FEEDS = [
    "http://rss.cnn.com/rss/edition_world.rss",                        # CNN World
    "https://feeds.npr.org/1004/rss.xml",                              # NPR World
]

YEMEN_FEEDS = [
    "https://reliefweb.int/updates/rss.xml?view=maps",                 # UN OCHA
    "https://reliefweb.int/updates/rss.xml",                           # UN OCHA Updates
    "https://rss.cnn.com/rss/edition_world.rss",                       # CNN World
    "https://feeds.npr.org/1004/rss.xml",                              # NPR World
]

# Add high-trust humanitarian feeds per your scope (fill known URLs you already used)
RELIEFWEB_FEEDS = {
    "ukraine": "...",  # keep placeholders if you prefer to paste later
    "syria":   "...",
    "yemen":   "...",
}
UNHCR_FEEDS = { "ukraine":"...", "syria":"...", "yemen":"..." }
WHO_FEEDS   = { "ukraine":"...", "syria":"...", "yemen":"..." }
REUTERS_FEEDS = {
    # You already have Google News RSS query for Reuters Ukraine; do analogous for others
    #"ukraine": "https://news.google.com/rss/search?q=intitle%3AUkraine+site%3Areuters.com&hl=en-US&gl=US&ceid=US%3Aen",
    #"syria":   "https://news.google.com/rss/search?q=intitle%3ASyria+site%3Areuters.com&hl=en-US&gl=US&ceid=US%3Aen",
    #"yemen":   "https://news.google.com/rss/search?q=intitle%3AYemen+site%3Areuters.com&hl=en-US&gl=US&ceid=US%3Aen",
    #"ukraine": "https://rsshub.app/reuters/world/europe",
    #"syria":   "https://rsshub.app/reuters/world/middle-east",
    #"yemen":   "https://rsshub.app/reuters/world/middle-east",
    "ukraine": "...",  # keep placeholders if you prefer to paste later
    "syria":   "...",
    "yemen":   "...",
}
BLOOMBERG_FEEDS = {
    # You already have Google News RSS query for Reuters Ukraine; do analogous for others
    #"ukraine": "https://www.bloomberg.com/authors/AQcJycm0xLU/aliaksandr-kudrytski.rss",
    #"syria":   "https://feeds.bloomberg.com/politics/news.rss",
    #"yemen":   "https://feeds.bloomberg.com/politics/news.rss",
    #"ukraine": "https://feeds.bloomberg.com/business/news.rss",  # keep placeholders if you prefer to paste later
    #"syria":   "https://feeds.bloomberg.com/business/news.rss",
    #"yemen":   "https://feeds.bloomberg.com/business/news.rss",
    "ukraine": "...",  # keep placeholders if you prefer to paste later
    "syria":   "...",
    "yemen":   "...",
}

COUNTRY_FEEDS = {
    "ukraine": UKRAINE_FEEDS + [REUTERS_FEEDS["ukraine"], RELIEFWEB_FEEDS["ukraine"], UNHCR_FEEDS["ukraine"], WHO_FEEDS["ukraine"], BLOOMBERG_FEEDS["ukraine"]],
    "syria":   SYRIA_FEEDS + [REUTERS_FEEDS["syria"],   RELIEFWEB_FEEDS["syria"],   UNHCR_FEEDS["syria"],   WHO_FEEDS["syria"],   BLOOMBERG_FEEDS["syria"]],
    "yemen":   YEMEN_FEEDS + [REUTERS_FEEDS["yemen"],   RELIEFWEB_FEEDS["yemen"],   UNHCR_FEEDS["yemen"],   WHO_FEEDS["yemen"],   BLOOMBERG_FEEDS["yemen"]],
}

KNOWN_DOMAINS = {
        "un.org": "United Nations News",
        "reuters.com": "Reuters",
        "bbc.com": "BBC",
        "apnews.com": "AP News",
        "nytimes.com": "New York Times",
        "washingtonpost.com": "Washington Post",
        "rferl.org": "RFE/RL",
        "un.org": "United Nations",
        "reliefweb.int": "ReliefWeb",
        "who.int": "WHO",
        "unhcr.org": "UNHCR",
        "wfp.org": "WFP",
        "kyivindependent.com": "Kyiv Independent",
        "cnn.com": "CNN",
        "abcnews.go.com": "ABC News",
        "cbsnews.com": "CBS News",
        "nbcnews.com": "NBC News",
        "bloomberg.com": "Bloomberg",
        "wsj.com": "Wall Street Journal",
        "washingtonpost.com": "Washington Post",
        "latimes.com": "LA Times",
        "politico.com": "Politico",
}

def guess_country(text_or_title: str) -> str | None:
    t = (text_or_title or "").lower()
    for c, kws in COUNTRY_KEYWORDS.items():
        if any(k in t for k in kws):
            return c
    return None
