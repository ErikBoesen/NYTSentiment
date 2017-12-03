import csv
from analysis import *

# Make request to NYT's API
raw = search(input('Enter search term: '))

# access the headlines and store them in a list
response = raw['response']['docs'][0]['headline']['main']
headlines = [[item['headline']['main']] for item in raw['response']['docs']]

# Write headlines to CSV file
with open('headlines.csv', 'w') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator='\n')
    for headline in headlines:
        writer.writerow(headline)

# analyze the sentiment of each headline given by the search term
analyze('headlines.csv')
