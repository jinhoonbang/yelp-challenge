import pandas as pd
import numpy as np
import os.path
import operator
from PIL import Image
import glob
import pickle

curr_dir = os.path.dirname(os.path.realpath(__file__))
print(curr_dir)
print('/'.join(("data", "pid_bid.csv")))
print('/'.join(("data", "yelp_academic_dataset_business.csv")))
path_pid = os.path.join(curr_dir, '/'.join(("data", "pid_bid.csv")))
path_business = os.path.join(curr_dir, '/'.join(("data", "yelp_academic_dataset_business.csv")))

##Parse business set
df = pd.read_csv(path_business, encoding='ISO-8859-1')
df = df[['business_id','categories']]

##Collect all classes (categories)
classes = []
categories = df['categories']
for cat in categories:
    #remove "[" and "]"
    cat = cat[1:-1]
    cat = cat.split(",")
    for item in cat:
        item = item.strip()[1:-1]
        if item not in classes:
            classes.append(item)

##Format DF so that [["business_id", list of "category"]]

list_cats = []
for cat in categories:
    cat = cat[1:-1]
    cat = cat.split(",")
    cats = []
    for item in cat:
        item = item.strip()[1:-1]
        cats.append(item)
    list_cats.append(cats)

print(len(df))
print(len(list_cats))

##Get list of all business id's
# list_bizy = []
# bizys = df['business_id']
# for bizy in bizys:
#     bizy = bizy[2:-1]
#     list_bizy.append(bizy)



##Assign numbers to classes
# m = {}
# counter = 0
# for cat in classes:
#     if cat not in m:
#         m[cat] = counter
#         counter += 1
##Initially 897 classes

##Get count of each category
m = {}
for cats in list_cats:
    for c in cats:
        if c not in m:
            m[c] = 1
        else:
            m[c] += 1

sorted_m = sorted(m.items(), key=operator.itemgetter(1))
sorted_m.reverse()
sorted_m = sorted_m[:20]

tags = []
for x in sorted_m:
    tags.append(x[0])

##Assign distinct number to each class (0 ~ 31)
m = {}
counter = 0
for t in tags:
    m[t] = counter
    counter += 1
print(m)

##Generate label column
##If an entry is two or more categories in top 32, then only use first one
label = []
for cats in list_cats:
    val = None
    for c in cats:
        if c in m:
            val = m[c]
            break
    label.append(val)

#Generate list pid
df_pid = pd.read_csv(path_pid, encoding='utf-8', engine='python')
df_pid = df_pid.drop('caption', axis = 1)

#Iterate through all images
# x = 0
# y = 0
# counter = 0

#Standard size = 410 x 390
# for pid in list_pid:
#     if pid is None:
#         print("none")
#     else:
#         path = os.path.join(curr_dir, "/".join(("data", "photos", pid + ".jpg")))
#         img = Image.open(path)
#         x += img.size[0]
#         y += img.size[1]
#         counter += 1

## Use Raw pixel as current feature vector

# base_width = 410
# base_height = 390

# pid = list_pid[0]
# path = os.path.join(curr_dir, "/".join(("data", "photos", pid + ".jpg")))
# img = Image.open(path)
# img = img.resize((base_width,base_height), Image.ANTIALIAS)
# pixels = list(img.getdata())
# width, height = img.size
# print(pixels)
# print(len(pixels))
# print(width)
# print(height)


print("label", len(label))
print("df", len(df))
#create a dataframe with label
df_bid = df.ix[:, 0]
df_label = pd.DataFrame(label, columns=['label'])

non_counter = 0
for row in label:
    if row is None:
        non_counter += 1

print("non_counter", non_counter)
df = pd.concat([df_bid, df_label], axis=1)
df = df.dropna()

path_business = os.path.join(curr_dir, '/'.join(("data","photos","*.jpg")))

photos = []
for path in glob.glob(path_business):
    photos.append(os.path.splitext(os.path.basename(path))[0])

##df_pid has business id and photo_id
df_pid = df_pid.loc[df_pid['photo_id'].isin(photos)]
df_pid = df_pid.drop_duplicates(subset='business_id', keep = "first")
df_bid = pd.Series(df["business_id"])
df_bid = df_bid.apply(lambda x: x[2:-1])
df_label = df.ix[:,1]

df = pd.concat([df_bid, df_label], axis=1)
df = pd.merge(df_pid, df, on='business_id')
df = df.drop('business_id', axis=1)

base_width = 50
base_height = 50

feature = []

counter = 0
for index, row in df.iterrows():
    print(counter)
    counter += 1
    curr = row["photo_id"]
    path = os.path.join(curr_dir, "/".join(("data", "photos", curr + ".jpg")))
    img = Image.open(path)
    img = img.resize((base_width,base_height), Image.ANTIALIAS)
    pixels = list(img.getdata())
    pixels = np.asarray(pixels).flatten()
    feature.append(pixels)

feature = np.asarray(feature)
print("feature.shape = {}", feature.shape)
label = df.ix[:,1].as_matrix()
label = np.reshape(label, (label.shape[0],1))
print("label.shape = {}", label.shape)
data = np.concatenate((label, feature), axis=1)
print(data)
print("data.shape = {}", data.shape)
dimension = np.array([data.shape[0], data.shape[1]])
data=np.append(dimension,data)
data.astype("float64")
data.tofile(os.path.join(curr_dir, "raw_data.bin"))

print(m)









