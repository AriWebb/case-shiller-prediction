import csv


def main():
    f = open('prop_crime.csv', 'w')
    writer = csv.writer(f)
    files = ['atlanta_1991_2020_all_prop.csv',
             'boston_1990_2020_all_prop.csv',
             'chicago_1990_2020_all_prop.csv',
             'cleveland_1990_2020_all_prop.csv',
             'dallas_2000_2020_all_prop.csv',
             'denver_1990_2020_all_prop.csv',
             'detroit_1991_2020_all_prop.csv',
             'la_1990_2020_all_prop.csv',
             'miami_1990_2020_all_prop.csv',
             'minneapolis_1990_2020_all_prop.csv',
             'ny_1987_2020_all_prop.csv',
             'phoenix_1987_2020_all_prop.csv',
             'portland_1990_2020_all_prop.csv',
             'sf_1990_2020_all_prop.csv',
             'seattle_1990_2020_all_prop.csv',
             'tampa_1990_2020_all_prop.csv',
             'dc_1996_2020_all_prop.csv']
    cities = ['atlanta', 'boston', 'chicago', 'cleveland', 'dallas', 'denver', 'detroit', 'la', 'miami',
              'minneapolis', 'nyc', 'phoenix', 'portland', 'sf', 'seattle', 'tampa', 'dc']
    city_starts = ['01-91', '01-90', '01-90', '01-90', '01-00', '01-90', '01-91', '01-90', '01-90', '01-90', '01-87',
                  '01-02', '01-90', '01-90', '01-90', '01-90', '11-96']
    assert len(files) == len(cities)
    assert len(city_starts) == len(cities)
    for i in range(len(files)):
        city = cities[i]
        for row in open(files[i]):
            num_cols = len(row.split(','))
        for j in range(1, num_cols):
            k = 0
            for row in open(files[i]):
                if k == 0:
                    year = row.split(',')[j]
                elif k == 1:
                    reported = row.split(',')[j].strip()
                elif k == 2:
                    cleared = row.split(',')[j].strip()
                k += 1
            for l in range(1, 13):
                date = str(l) + '-' + year[2:].strip()
                if l < 10:
                    date = '0' + date
                datapoint = [city, date, reported]
                writer.writerow(datapoint)
    f.close()


if __name__ == '__main__':
    main()