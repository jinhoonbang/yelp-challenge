import pandas as pd
import glob

# PATH = '/Users/jinbang-mac/development/yelp-challenge/data/yelp_dataset_challenge_academic_dataset/*.json'

# for file in glob.glob(PATH):
#     print(file)

PATH_BUSINESS = '/Users/jinbang-mac/development/yelp-challenge/data/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'

#file://localhost/path/to/table.json

# read the entire file into a python array
with open(PATH_BUSINESS, 'r') as f:
    data = f.readlines()

# remove the trailing "\n" from each line
data = map(lambda x: x.rstrip(), data)

# each element of 'data' is an individual JSON object.
# i want to convert it into an *array* of JSON objects
# which, in and of itself, is one large JSON object
# basically... add square brackets to the beginning
# and end, and have all the individual business JSON objects
# separated by a comma
data_json_str = "[" + ','.join(data) + "]"

# now, load it into pandas
data_df = pd.read_json(data_json_str)

print(data_df)
