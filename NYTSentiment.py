import csv
from SentimentIngress import *

searchTerm = input('What do you want to analyze?')

# Make request to NYT's API
dataSetRaw = makeJsonRequest(searchTerm)

# access the headlines and store them in a list
response = dataSetRaw['response']['docs'][0]['headline']['main']
headlines = []
for item in dataSetRaw['response']['docs']:
    headlines.append([item['headline']['main']])

# write them into a csv file
with open('headlines.csv', 'w') as headlineFile:
    writer = csv.writer(headlineFile, quoting=csv.QUOTE_ALL,
                        lineterminator='\n')
    for headline in headlines:
        writer.writerow(headline)

# analyze the sentiment of each headline given by the search term
AnalyzeSentiment('headlines.csv')
