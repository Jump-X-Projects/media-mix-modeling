# config.py

# --- Data ---
DATE_COLUMN = "Date"
DATE_FORMAT = {
    "YYYY-MM-DD": "%Y-%m-%d",
    "MM/DD/YYYY": "%m/%d/%Y",
    "DD/MM/YYYY": "%d/%m/%Y",
}

# --- Media Channels (replace with your actual channels) ---
MEDIA_SPEND_COLUMNS = [
    "TV",
    "Radio",
    "Banners",
    "search_clicks_PPC",
    "facebook_I",
    "newsletter",
]
NON_MEDIA_COLUMNS = ["Easter", "Christmas", "Xmas"]
MEDIA_CHANNELS = ["TV", "Radio", "Banners"]
TARGET_VARIABLE = "Sales"
COSTS = {
    "TV": 1000,
    "Radio": 500,
    "Banners": 100,
    "search_clicks_PPC": 200,
    "facebook_I": 300,
    "newsletter": 50,
}