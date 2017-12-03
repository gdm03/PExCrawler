import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class CSVOperations():
    def string_to_datetime(self, s):
        return datetime.strptime(s, "%d-%b-%y").date()

    def get_daterange(self, li):
        print(min(li), max(li))

    # Filters most recent threads by most recent dates
    def filter_dates(self, dates, limit):
        return sorted(dates, key=lambda x: datetime.strptime(x[2], "%d-%b-%y"), reverse=True)[:limit]

    # 1st graph: x - time, y - number of posts
    def plot_number_of_posts(self):
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
                #df_post_content = new_df['post_content'].loc[df['thread_title'] == thread].tolist()
                #print(df_post_content)

                if len(thread_list) > THREAD_LIMIT:
                    temp_list.append([subforum, thread, max(df_post_times)])
                    #print(temp_list)

                else:
                    recent_threads.append([subforum, thread, max(df_post_times)])

            if temp_list:
                filtered_list = self.filter_dates(temp_list, THREAD_LIMIT)

                for item in filtered_list:
                    recent_threads.append(item)

        df_temp = pd.DataFrame(recent_threads, columns=['subforum', 'thread_title', 'post_time'])
        temp_list2 = []

        for thread in df_temp['thread_title']:
            #print(thread)
            df_temp2 = df[['subforum', 'thread_title', 'post_content', 'post_time']].loc[df['thread_title'] == thread]
            temp_list2.append(df_temp2)
            #print(df_recent_threads)
            print("------------------------------------------")

        #df_recent_threads = pd.concat(temp_list2)
        df_recent_threads = pd.concat(temp_list2, ignore_index=True)

        df_recent_threads['counts'] = df_recent_threads.groupby(['subforum', 'thread_title', 'post_time'])[
            'post_time'].transform('count')

        #groups per subforum and thread_title (for graphing)
        #print(df_recent_threads.head(500))
        df_recent_threads['post_time'] = pd.to_datetime(df_recent_threads['post_time'])
        grouped = df_recent_threads.groupby(['subforum', 'thread_title'])
        fig, ax = plt.subplots(figsize=(8, 6))

        for label, tdf in grouped:
            print(tdf[['subforum', 'thread_title', 'post_time', 'counts']])
            #tdf['post_time'] = pd.to_datetime(tdf['post_time'])
            tx = tdf.plot(x='post_time', y='counts', kind='line', ax=ax, label=label[1])
            print("-----------------")
            tx.set(xlabel='Post date', ylabel='Number of posts')
        plt.legend()
        plt.show()


        # grouped = df_recent_threads.groupby(['subforum', 'thread_title'])
        # fig, ax = plt.subplots(figsize=(8, 6))
        # for label, tdf in grouped:
        #     tdf.plot(x='post_time', y='counts', kind='line', ax=ax, label=label)
        # plt.legend()
        # plt.show()


        # classes = ["class 1"] * 5 + ["class 2"] * 5 + ["class 3"] * 5
        # vals = [datetime.strptime('Jun 1 2005', '%b %d %Y'),
        #         datetime.strptime('Jun 3 2005', '%b %d %Y'),
        #         datetime.strptime('Jun 5 2005', '%b %d %Y'),
        #         datetime.strptime('Jun 6 2005', '%b %d %Y'),
        #         datetime.strptime('Jun 8 2006', '%b %d %Y')] + \
        #        [datetime.strptime('Jul 1 2005', '%b %d %Y'),
        #         datetime.strptime('Jul 3 2005', '%b %d %Y'),
        #         datetime.strptime('Jul 5 2005', '%b %d %Y'),
        #         datetime.strptime('Jul 6 2005', '%b %d %Y'),
        #         datetime.strptime('Jul 8 2005', '%b %d %Y')] + \
        #        [datetime.strptime('Jul 1 2006', '%b %d %Y'),
        #         datetime.strptime('Jul 3 2006', '%b %d %Y'),
        #         datetime.strptime('Jul 5 2006', '%b %d %Y'),
        #         datetime.strptime('Jul 6 2006', '%b %d %Y'),
        #         datetime.strptime('Jul 8 2006', '%b %d %Y')]
        #
        # p_df = pd.DataFrame({"class": classes, "vals": vals})
        #
        # fig, ax = plt.subplots(figsize=(8, 6))
        #
        # for label, tdf in p_df.groupby('class'):
        #     print(tdf)
        #     tdf.reset_index().plot(x='vals', y='index', kind="line", ax=ax, label=label)
        #
        # plt.legend()
        # plt.show()


    def compute_post_depth(self):
        df = pd.read_csv('sorted.csv')

        thread_list = np.unique(df['thread_title'])
        final = pd.DataFrame()

        for thread in thread_list:
            # Get only post_content and quoted_post columns from original csv
            #new_df = df[['post_content', 'quoted_post']].loc[df['thread_title'] == thread]
            new_df = df.loc[df['thread_title'] == thread]
            # Add column depth and initialize values to 0
            new_df['depth'] = 0

            # Check all non-empty quoted_posts
            qp_indices = new_df.index[new_df['quoted_post'].notnull()]

            for idx in qp_indices:
                # Get index of matching quoted_post to post_content
                match = new_df.index[new_df['post_content'] == new_df['quoted_post'].loc[idx]].tolist()

                if match:
                    new_df['depth'].loc[idx] = new_df['depth'].loc[match[0]] + 1

                else:
                    if ',<sep>,' in new_df['quoted_post'].loc[idx]:
                        strings = new_df['quoted_post'].loc[idx].split(',<sep>,')
                        depth_list = []

                        for current_string in strings:
                            new_idx = new_df.index[new_df['post_content'] == current_string].tolist()

                            if new_idx:
                                depth_list.append(new_df['depth'].loc[new_idx[0]] + 1)

                        # Auto do nothing if depth_list is null
                        new_df['quoted_post'].loc[idx] = ', '.join(x for x in strings)
                        new_df['depth'].loc[idx] = ', '.join(str(x) for x in depth_list)

            final = final.append(new_df)

        final.to_csv('final.csv', encoding='utf-8', index=False)

if __name__ == '__main__':
    #CSVOperations().compute_post_depth()
    #CSVOperations().plot_post_gap()
    CSVOperations().plot_number_of_posts()
