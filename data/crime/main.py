import csv


def main():
    f = open('crime.csv', 'w')
    writer = csv.writer(f)
    files = ['Summary-Crime-reported-by-the-Atlanta-Police-Department-1991-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-Boston-Police-Department-1990-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-Chicago-Police-Department-1990-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-Cleveland-Police-Department-1990-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-Dallas-Police-Department-2000-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-Denver-Police-Department-1990-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-Detroit-Police-Department-1991-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-Los-Angeles-Police-Department-1990-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-Miami-Police-Department-1990-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-Minneapolis-Police-Department-1990-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-New-York-City-Police-Department-1987-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-Phoenix-Police-Department-1987-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-Portland-Police-Department-1990-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-San-Francisco-Police-Department-1990-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-Seattle-Police-Department-1990-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-Tampa-Police-Department-1990-20205_5_2022.csv',
             'Summary-Crime-reported-by-the-Washington-Police-Department-1996-20205_5_2022.csv']
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
                datapoint = [city, date, reported, cleared]
                writer.writerow(datapoint)
    f.close()


if __name__ == '__main__':
    main()