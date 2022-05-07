import csv

cities = ['atlanta', 'boston', 'chicago', 'cleveland', 'dallas', 'denver', 'detroit', 'la', 'miami',
          'minneapolis', 'nyc', 'phoenix', 'portland', 'sf', 'seattle', 'tampa', 'dc']

LABELS = ['atlanta', 'boston', 'chicago', 'cleveland', 'dallas', 'denver', 'detroit', 'la', 'miami',
          'minneapolis', 'nyc', 'phoenix', 'portland', 'sf', 'seattle', 'tampa', 'dc', 'date', 'cpi', 'crimes_reported', 'crimes_cleared', 'patents', 'population', 'unemployment',
          'case_shiller', 'dow', 'nasdaq', 'sp', 'label']

LABELS_NO_DATE = ['atlanta', 'boston', 'chicago', 'cleveland', 'dallas', 'denver', 'detroit', 'la', 'miami',
          'minneapolis', 'nyc', 'phoenix', 'portland', 'sf', 'seattle', 'tampa', 'dc', 'cpi', 'crimes_reported', 'crimes_cleared', 'patents', 'population', 'unemployment',
          'case_shiller', 'dow', 'nasdaq', 'sp', 'label']


def main():
    city_files = ['cpi/cpi.csv', 'crime/crime.csv', 'Patents_processed.csv',
                  'Population_processed.csv', 'unemployment_processed.csv', 'Case_Shiller_processed.csv']
    stock_files = ['Processed_Stock_Data/DOW_processed.csv', 'Processed_Stock_Data/NASDAQ_processed.csv',
                   'Processed_Stock_Data/SP_processed.csv']

    # Create Case_Shiller dictionary
    case_shiller_dict = {}
    for row in open('Case_Shiller_processed.csv'):
        row = row.strip().split(',')
        city_date = row[0] + ',' + row[1]
        case_shiller_dict[city_date] = row[2]

    # Combine all city data
    all_data = {}
    for file in city_files:
        for row in open(file):
            row = row.strip().split(',')
            key = row[0] + ',' + row[1]
            if key not in all_data:
                all_data[key] = [value for value in row[2:]]
            else:
                for value in row[2:]:
                    all_data[key].append(value)

    # Keep examples with all features and label
    num_features = max([len(lst)] for lst in all_data.values())[0]
    full_data = {}  # Dictionary with data points that have all features
    for key in all_data:
        city = key.split(',')[0]
        future_month_str = key.split(',')[1][:3]
        future_year_int = int(key.split(',')[1][3:]) + 5
        if future_year_int >= 100:
            future_year_str = str(future_year_int)[1:]
        else:
            future_year_str = str(future_year_int)
        future_date = future_month_str + future_year_str
        # If we have the label and all the features
        future_key = city + ',' + future_date  # Date five years in the future
        if future_key in case_shiller_dict.keys() and len(all_data[key]) == num_features:
            full_data[key] = all_data[key]
            full_data[key].append(case_shiller_dict[future_key])

    # Add stock data
    for key in full_data:
        date = key.split(',')[1]
        for file in stock_files:
            for row in open(file):
                if row.strip().split(',')[0] == date:
                    full_data[key].append(row.strip().split(',')[1])
                    break

    # Move label to last entry
    for key in full_data:
        full_data[key].append(full_data[key].pop(-4))

    #Make a dictionary mapping city names to one-hot representations
    #one-hot representations are based on cities list
    d_one_hot = {}
    one = '1'
    zero = '0'
    for i in range(len(cities)):
        cur = []
        for j in range(len(cities)):
            if j == i:
                cur += one
            else:
                cur += zero
        d_one_hot[cities[i]] = cur

    # Write to csv
    #for windows I need to have the newline=''
    f = open('all_data_w_city_names.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(LABELS)
    for key in full_data:
        data = []
        for i in d_one_hot[key.split(',')[0]]:
            data.append(i)
        data.append(key.split(',')[1])
        for value in full_data[key]:
            data.append(value)
        assert (len(data) == len(LABELS))
        writer.writerow(data)
    f.close()

    f = open('all_data_w_city_names_no_date.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(LABELS_NO_DATE)
    for key in full_data:
        data = []
        for i in d_one_hot[key.split(',')[0]]:
            data.append(i)

        for value in full_data[key]:
            data.append(value)
        assert (len(data) == len(LABELS_NO_DATE))
        writer.writerow(data)
    f.close()


if __name__ == '__main__':
    main()