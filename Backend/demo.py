import sentiment_analysis_module

url = "https://www.allsides.com/story/general-news-thanksgiving-travel-season-gas-prices-weather-delays-and-more" 
results = sentiment_analysis_module.analyze_url(url)
print(results)