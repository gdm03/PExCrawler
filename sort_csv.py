import csv

def sort_data():
    with open('output.csv', 'r', encoding='utf8') as unsorted:
        reader = csv.reader(unsorted)
        header = next(reader, None)
        # 1 - thread_title, 3 - post_counter
        sortedlist = sorted(reader, key=lambda i: (i[1], int(i[3])), reverse=False)

    with open('sorted.csv', 'w', newline='', encoding='utf8') as output:
        wr = csv.writer(output, dialect='excel')
        if header:
            wr.writerow(header)
        wr.writerows(sortedlist)

if __name__ == '__main__':
    sort_data()
