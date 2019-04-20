import json

d = []

file_name = 'config.json'
with open(file_name) as f:
    json_file = [line.rstrip('\n') for line in f]
    json_file = ''.join(json_file)
    d = json.loads(json_file)

rootDir = d['rootDir']    
metadata= d['metaData']
url = d['url']
labels = d['labels']
