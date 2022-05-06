import csv


def main():
    f = open('cpi.csv', 'w')
    writer = csv.writer(f)
    files = ['atlanta_bi.csv', 'atlanta_monthly.csv', 'atlanta_semi.csv', 'boston_bi.csv', 'chicago_monthly.csv',
             'cleveland_bi.csv', 'dallas_bi.csv', 'dc_bi.csv', 'denver_semi.csv', 'detroit_bi.csv', 'la_monthly.csv',
             'miami_bi.csv', 'minneapolis_semi.csv', 'nyc_monthly.csv', 'phoenix_semi.csv', 'portland_semi.csv',
             'seattle_bi.csv', 'seattle_semi.csv', 'seattle_monthly.csv', 'sf_bi.csv', 'tampa_semi.csv',
             'tampa_annual.csv', 'tampa_monthly.csv']
    cities = ['atlanta', 'boston', 'chicago', 'cleveland', 'dallas', 'denver', 'detroit', 'la', 'miami',
              'minneapolis', 'nyc', 'phoenix', 'portland', 'sf', 'seattle', 'tampa', 'dc']
    city_starts = ['01-91', '01-90', '01-90', '01-90', '01-00', '01-90', '01-91', '01-90', '01-90', '01-90', '01-87',
                  '01-02', '01-90', '01-90', '01-90', '01-90', '11-96']
    assert len(city_starts) == len(cities)
    for i in range(len(cities)):
        city = cities[i]
        city_files = [file for file in files if city + '_' in file]
        for file in city_files:
            if '_bi' in file:
                last_num = 0
                first_row = True
                for row in open(file):
                    if first_row:
                        first_row = False
                    else:
                        date = row.split(',')[0][5:7] + '-' + row.split(',')[0][2:4]
                        if not last_num:
                            last_num = float(row.split(',')[1].strip())
                            datapoint = [city, date, last_num]
                            writer.writerow(datapoint)
                        elif row.split(',')[1].strip() == '.':
                            mid_date = date
                        elif row.split(',')[1].strip() != '.':
                            cur_num = float(row.split(',')[1].strip())
                            mid_cpi = (last_num + cur_num) / 2
                            writer.writerow([city, mid_date, mid_cpi])
                            writer.writerow([city, date, cur_num])
                            last_num = cur_num
            elif '_monthly' in file:
                first_row = True
                for row in open(file):
                    if first_row:
                        first_row = False
                    else:
                        date = row.split(',')[0][5:7] + '-' + row.split(',')[0][2:4]
                        cpi = float(row.split(',')[1].strip())
                        datapoint = [city, date, cpi]
                        writer.writerow(datapoint)
            elif '_annual' in file:
                last_num = 0
                first_row = True
                for row in open(file):
                    if first_row:
                        first_row = False
                    else:
                        date = row.split(',')[0][5:7] + '-' + row.split(',')[0][2:4]
                        if not last_num:
                            last_num = float(row.split(',')[1].strip())
                            last_date = date
                        else:
                            cur_num = float(row.split(',')[1].strip())
                            for z in range(1, 13):
                                num_to_write = last_num + (cur_num - last_num)*(z - 1)/12
                                date_to_write = str(z) + last_date[2:]
                                if z < 10:
                                    date_to_write = '0' + date_to_write
                                writer.writerow([city, date_to_write, num_to_write])
                            last_num = cur_num
                            last_date = date
                writer.writerow([city, last_date, last_num])
            elif '_semi' in file:
                last_num = 0
                first_row = True
                for row in open(file):
                    if first_row:
                        first_row = False
                    else:
                        date = row.split(',')[0][5:7] + '-' + row.split(',')[0][2:4]
                        if not last_num:
                            last_num = float(row.split(',')[1].strip())
                            last_date = date
                        else:
                            cur_num = float(row.split(',')[1].strip())
                            last_month = int(last_date[:2])
                            for z in range(1, 7):
                                num_to_write = last_num + (cur_num - last_num) * (z - 1) / 6
                                if last_month == 1:
                                    date_to_write = str(z) + last_date[2:]
                                else:
                                    date_to_write = str(z + 6) + last_date[2:]
                                if z < 10:
                                    date_to_write = '0' + date_to_write
                                writer.writerow([city, date_to_write, num_to_write])
                            last_num = cur_num
                            last_date = date
                writer.writerow([city, last_date, last_num])
    f.close()


if __name__ == '__main__':
    main()