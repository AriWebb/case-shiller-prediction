import csv

LABELS = ['city', 'date', 'cpi', 'crimes_reported', 'crimes_cleared', 'patents', 'population', 'case_shiller', 'dow',
          'nasdaq', 'sp']

def main():
    all_data = {}
    city_files = ['cpi/cpi.csv', 'crime/crime.csv', 'Patents_processed.csv',
                  'Population_processed.csv', 'Case_Shiller_processed.csv']
    stock_files = ['Processed_Stock_Data/DOW_processed.csv', 'Processed_Stock_Data/NASDAQ_processed.csv',
                   'Processed_Stock_Data/SP_processed.csv']
    for file in city_files:
        first_row = True
        for row in open(file):
            if first_row:
                first_row = False
            else:
                row = row.strip().split(',')
                key = row[0] + ',' + row[1]
                if key not in all_data:
                    all_data[key] = [value for value in row[2:]]
                else:
                    for value in row[2:]:
                        all_data[key].append(value)

    num_features = max([len(lst)] for lst in all_data.values())[0]
    full_data = {}  # Dictionary with data points that have all features
    for key in all_data:
        if len(all_data[key]) == num_features:
            full_data[key] = all_data[key]
    for key in full_data:
        date = key.split(',')[1]
        for file in stock_files:
            for row in open(file):
                if row.strip().split(',')[0] == date:
                    full_data[key].append(row.strip().split(',')[1])
                    break
    f = open('all_data_w_city_names.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(LABELS)
    for key in full_data:
        data = [key.split(',')[0], key.split(',')[1]]
        for value in full_data[key]:
            data.append(value)
        assert(len(data) == len(LABELS))
        writer.writerow(data)
    f.close()


if __name__ == '__main__':
    main()