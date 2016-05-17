import csv
import simplejson as json
import pandas as pd

json_file = "/Users/jinbang-mac/development/yelp-challenge/data/2016_yelp_dataset_challenge_photos/photo_id_to_business_id.json"

csv_file = '{0}.csv'.format(json_file.split('.json')[0])
columns = []
#photo_id":"zzztSJpg-6_ovI0KXjAwuw","business_id":"50SlLpT5bC5w2MU_DS62Fg","caption":"","label":"none
m = {}

# with open(json_file) as fin:
#     data = fin.readlines()[0]
#     #remove "[{", "}]" at the beginning and end of string
#     data = data[2:-2]
#     data = data.split("},{")

#     #generate columns
#     for line in data:
#         #remove quotation at beginning and end
#         line = line[1:-1]
#         entries = line.split('","')
#         for entry in entries:
#             print(entry)

with open(json_file) as fin:
    data = fin.readlines()[0]
    jsons = json.loads(data)
    df = pd.DataFrame(jsons)
    df = df.drop('label', 1)
    print(df)

    df.to_csv("data/pid_bid.csv", index=False, encoding='utf-8')







