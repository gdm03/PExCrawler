import csv
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

class CSVOperations():
    # Filters most recent threads by most recent dates
    def filter_dates(self, dates, limit):
        return sorted(dates, key=lambda x: datetime.strptime(x[2], "%d-%b-%y"), reverse=True)[:limit]

    def plot_post_depth(self):
        pass

    # Update to pandas sort, take note of encoding
    def sort_data(self):
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

    # misleading function, change later
    def compute_post_time_counts(self):
        THREAD_LIMIT = 5

        df = pd.read_csv('sorted.csv')
        subforum_list = np.unique(df['subforum'])

        recent_threads = []

        for subforum in subforum_list:
            new_df = df[['thread_title', 'post_time']].loc[df['subforum'] == subforum]
            thread_list = np.unique(new_df['thread_title'])

            temp_list = []

            for thread in thread_list:
                df_post_times = new_df['post_time'].loc[df['thread_title'] == thread].tolist()

                if len(thread_list) > THREAD_LIMIT:
                    temp_list.append([subforum, thread, max(df_post_times)])

                else:
                    recent_threads.append([subforum, thread, max(df_post_times)])

            if temp_list:
                filtered_list = self.filter_dates(temp_list, THREAD_LIMIT)

                for item in filtered_list:
                    recent_threads.append(item)

        df_temp = pd.DataFrame(recent_threads, columns=['subforum', 'thread_title', 'post_time'])
        temp_list2 = []

        for thread in df_temp['thread_title']:
            df_temp2 = df[['subforum', 'thread_title', 'post_content', 'post_time']].loc[df['thread_title'] == thread]
            temp_list2.append(df_temp2)
            print("------------------------------------------")

        df_recent_threads = pd.concat(temp_list2, ignore_index=True)

        df_recent_threads['counts'] = df_recent_threads.groupby(['subforum', 'thread_title', 'post_time'])[
            'post_time'].transform('count')

        return df_recent_threads

    # Plots the number of posts: x - time, y - number of posts
    def plot_number_of_posts(self, df):
        #groups per subforum and thread_title (for graphing)

        df['post_time'] = pd.to_datetime(df['post_time'])
        grouped = df.groupby('subforum')

        for label, tdf in grouped:
            grouped2 = tdf.groupby('thread_title')
            fig, ax = plt.subplots(figsize=(8, 6))
            for label2, tdf2 in grouped2:
                if tdf2['post_time'].nunique() == 1 and tdf2['counts'].nunique() == 1:
                    tx = tdf2.plot(x='post_time', y='counts', kind='line', marker='o', ax=ax, label=label2, lw=2, linestyle='None')
                else:
                    tx = tdf2.plot(x='post_time', y='counts', kind='line', ax=ax, label=label2, lw=2)

            tx.set(xlabel='Post date', ylabel='Number of posts')
            plt.title(label)
            plt.legend()
            plt.show()

    # Computes depth of post based on quoted posts
    def compute_post_depth(self):
        df = pd.read_csv('sorted.csv')

        thread_list = np.unique(df['thread_title'])
        final = pd.DataFrame()

        for thread in thread_list:
            # Get only post_content and quoted_post columns from original csv
            new_df = df.loc[df['thread_title'] == thread]
            # Add column depth and initialize values to 0
            new_df['depth'] = 0
            # print(new_df)

            # Check all non-empty quoted_posts
            qp_indices = new_df.index[new_df['quoted_post'].notnull()]

            for idx in qp_indices:
                # Get index of matching quoted_post to post_content
                match = new_df.index[new_df['post_content'] == new_df['quoted_post'].loc[idx]].tolist()

                # if there is quoted_post match
                if match:
                    # if there are multiple depths, (e.g. "1, 1")
                    # if type(new_df['depth'].loc[match[0]]) is str:
                    #     new_df['depth'].loc[idx] = 1
                    if type(new_df['depth'].loc[match[0]]) is str:
                        if ',<sep>,' in new_df['quoted_post'].loc[match[0]]:
                            strings = new_df['quoted_post'].loc[match[0]].split(',<sep>,')

                            for current_string in strings:
                                new_idx = new_df.index[new_df['post_content'] == current_string].tolist()

                                if new_idx:
                                    new_df['depth'].loc[idx] = new_df['depth'].loc[new_idx[0]] + 1

                                print(idx, match[0], "curr: ", current_string)
                        else:
                            depths = new_df['depth'].loc[match[0]].split(', ')
                            strings = new_df['quoted_post'].loc[match[0]].split(',<sep>,')

                            for x in depths:
                                new_df['quoted_post'].loc[idx]
                    else:
                        new_df['depth'].loc[idx] = new_df['depth'].loc[match[0]] + 1

                # if there is no quoted post match
                else:
                    if ',<sep>,' in new_df['quoted_post'].loc[idx]:
                        strings = new_df['quoted_post'].loc[idx].split(',<sep>,')
                        depth_list = []

                        for current_string in strings:
                            new_idx = new_df.index[new_df['post_content'] == current_string].tolist()

                            if new_idx:
                                depth_list.append(new_df['depth'].loc[new_idx[0]] + 1)

                        # Auto do nothing if depth_list is null
                        # new_df['quoted_post'].loc[idx] = ', '.join(x for x in strings)
                        # new_df['quoted_post'].loc[idx] = ' '.join(x for x in strings)
                        new_df['depth'].loc[idx] = ', '.join(str(x) for x in depth_list)


            final = final.append(new_df)

        final.to_csv('final.csv', encoding='utf-8', index=False)

    def compute_post_gap(self):
        df = pd.read_csv('pex.csv')
        post_dates = df['post_time']
        df['post_gap'] = 0
        # df['post_gap'] = pd.to_datetime(df['post_time']) - pd.to_datetime(df['post_time']).shift(-1)
        final = pd.DataFrame()

        # for i, (index, row) in enumerate(df.iterrows()):
        for index, row in df.iterrows():
            # current_time = pd.to_datetime(row['post_time'])
            print(index)
            if index != 0:
                current_time = pd.to_datetime(df['post_time'].loc[index - 1])

            else:
                current_time = 0

            if row['post_counter'] == 1:
                post_gap_int = 0
                # print(row['post_counter'], row['post_time'], post_gap_int)
            else:
                # post_gap_dt = pd.to_datetime(row['post_time']) - pd.to_datetime(current_time)
                post_gap_dt = pd.to_datetime(row['post_time']) - current_time
                # print(row['post_counter'], row['post_time'], gap)
                post_gap_int = (post_gap_dt / np.timedelta64(1, 'D')).astype(int)
                # print(row['thread_title'], row['post_counter'], row['post_time'], post_gap_dt, post_gap_int, type(row['post_counter']))

            # print(post_gap_int)
            df['post_gap'].loc[index] = post_gap_int

        # print(df['post_gap'])

        final = final.append(df)
        final.to_csv('final.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    # CSVOperations().compute_post_depth()
    #CSVOperations().plot_post_gap()
    CSVOperations().compute_post_gap()
    #CSVOperations().sort_data()
    # CSVOperations().compute_post_gap()
    # CSVOperations().plot_number_of_posts(CSVOperations().compute_post_time_counts())
