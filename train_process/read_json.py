import os
import xml.etree.ElementTree as ET
import json

# categories = ['', '']
#
file_path = r'D:\Myapps\anylabeling\fish\1\image00000.json'
#
# data = json.load(file_path)
# print(data)

info = {'name':'Tollr',
               'age':10,
               'statue':'student'
        }

with open(file_path, 'r') as json_file:

    json_Data = json.load(json_file)
for i in json_Data['shapes']:
    if i['shape_type'] == 'rectangle':
        print(i)
# print(json_Data['shapes'][0])
