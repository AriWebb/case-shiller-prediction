import csv

LABELS = ['city', 'date', 'cpi', 'crimes_reported', 'crimes_cleared', 'patents', 'population', 'case_shiller', 'dow',
          'nasdaq', 'sp', 'label']

def main():
    all_data = {}
    city_files = ['cpi/cpi.csv', 'crime/crime.csv', 'Patents_processed.csv',
                  'Population_processed.csv', 'Case_Shiller_processed.csv']
    stock_files = ['Processed_Stock_Data/DOW_processed.csv', 'Processed_Stock_Data/NASDAQ_processed.csv',
                   'Processed_Stock_Data/SP_processed.csv']

    # Create Case_Shiller dictionary
    case_shiller_dict = {}
    for row in open('Case_Shiller_processed.csv'):
        row = row.strip().split(',')
        city_date = row[0] + ',' + row[1]
        case_shiller_dict[city_date] = row[2]

    # Combine all city data
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
        future_key = city + ',' + future_date
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

    # Write to csv
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