import csv
import os
import re

input_dir = './input'
output_dir = './output'
for filename in os.listdir(input_dir):
    in_path = os.path.join(input_dir, filename)
    print(in_path)
    with open(in_path, newline='') as file:
        print(filename)
        city = re.sub('\.csv$', '', filename)
        print(city)
        out_path = os.path.join(output_dir, filename)
        out = open(out_path, 'w', newline='')
        out_writer = csv.writer(out)
        next(file)
        for line in file:
            li = line.split(',')
            row = [city, li[2][-2:] + '-' + li[1][-2:], li[4][:-1]]
            out_writer.writerow(row)
        out.close()
cities = ['atlanta', 'boston', 'chicago', 'cleveland', 'dallas', 'denver', 'detroit', 'la', 'miami',
          'minneapolis', 'nyc', 'phoenix', 'portland', 'sf', 'seattle', 'tampa', 'dc']
city_start = ['01-91', '01-90', '01-90', '01-90', '01-00', '01-90', '01-91', '01-90', '01-90', '01-90', '01-87',
              '01-02', '01-90', '01-90', '01-90', '01-90', '11-96']
start_dic = {}
i = 0
for city in cities:
    start_dic[city] = city_start[i]
    i += 1
with open('../unemployment_processed.csv', 'w', newline='') as out:
    for city in cities:
        in_path = os.path.join(output_dir, city + '.csv')
        file = open(in_path, newline='')
        reader = csv.reader(file)
        past_line = 0
        for line in reader:
            if line[1] == start_dic[city]:
                past_line = reader.line_num
                break
        num = 372
        file.close()
        file = open(in_path, newline='')
        reader = csv.reader(file)
        out_writer = csv.writer(out)
        for i, row in enumerate(reader):
            if i in range(past_line-1, num):
                out_writer.writerow(row)
        file.close()


