import numpy as np
import pandas as pd
from itertools import chain
import re

class CSVOperations():
    def remove_duplicates(self, seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def check_substring(self):
        print()

    def compute_depth(self):
        df = pd.read_csv('sorted.csv')

        thread_list = np.unique(df['thread_title'])

        for thread in thread_list:
            new_df = df[['post_content', 'quoted_post']].loc[df['thread_title'] == thread]
            new_df['depth'] = 0

            qp_indices = new_df.index[new_df['quoted_post'].notnull()]

            for idx in qp_indices:
                match = new_df.index[new_df['post_content'] == new_df['quoted_post'].loc[idx]].tolist()
                if match:
                    new_df['depth'].loc[idx] = new_df['depth'].loc[match[0]] + 1
                else:
                    if ',<sep>,' in new_df['quoted_post'].loc[idx]:
                        strings = new_df['quoted_post'].loc[idx].split(',<sep>,')

                        test_list = []

                        for i in strings:
                            match2 = new_df.index[new_df['post_content'] == i].tolist()
                            print(match2)

                            if match2:
                                test_list.append(new_df['depth'].loc[match2[0]] + 1)
                                print(test_list)
                                # new_df['depth'].loc[idx] = new_df['depth'].loc[match2[0]] + 1
                        new_df['depth'].loc[idx] = ', '.join(str(x) for x in test_list)

                    match.append('check')

                #match_indices.append(match)

            #flat = list(chain.from_iterable(match_indices))
            print(new_df)

            #print(flat)

        # indices of quoted post's that are not null, reference using qp_idx[0]
        qp_idx = np.array(np.where(pd.notnull(df['quoted_post']))).tolist()

        # list of quoted_post's indices with respective thread_title
        # [0] - index [1] - thread title
        threads = df.loc[qp_idx[0], 'thread_title'].tolist()
        qp_list = [list(a) for a in zip(qp_idx[0], threads)]

        depth = 0
        #matches = df.loc[df['quoted_post'].isin(df['post_content']) | ',<sep>,' in df['quoted_post']]
        #print(df['quoted_post'].isin(df['post_content']))
        #test = df.index[df['quoted_post'].isin(df['post_content'])].tolist()
        # print(df[['quoted_post']].merge(df[['quoted_post', 'post_content']], on='quoted_post',how='left'))
        # for x in test:
        #     print(x)
            # print(df['quoted_post'].loc[x])

        #filtered_matches = matches[['post_content', 'quoted_post']]

        #print(filtered_matches.merge(df['post_content'], on='post_content', how='left'))
        # with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
        #     print(filtered_matches)

        # for qp in qp_list:
        #     current_post = df['quoted_post'].loc[qp[0]]
        #     df_current_post = pd.Series(current_post)
        #     posts_in_current_thread = df.loc[df['thread_title'] == qp[1], 'post_content']
        #     matches = df_current_post.isin(posts_in_current_thread)
        #
        #
        #     #if df_current_post.isin(posts_in_current_thread).any():
        #     #print(posts_in_current_thread.index)
        #
        #     #df.merge(mdf, on='id', how='left')
        #     for match in matches:
        #         if match:
        #             pass
        #         else:
        #             if ',<sep>,' in current_post:
        #                 #print("2+ quotes: ", current_post)
        #                 multi_quotes = pd.Series(current_post.split(',<sep>,'))
        #                 for post in multi_quotes.isin(posts_in_current_thread):
        #                     if post:
        #                         depth += 1

            #print('------------------------------------------------------------------------------------')




        # lookup and match qp_idx to thread_name, iterate over range and look for post_content == quoted_post
        # for thread in unique_threads:
        #     if thread in threads_name:
        #         for i in threads_idx[:-1]:
                    #if df['quoted_post'].iloc[qp_idx[i]] in df['post_content'].iloc[threads_idx[i]:threads_idx[i+1]]:
                    # if df['post_content'].iloc[threads_idx[i]:threads_idx[i+1]].str.contains(df['quoted_post'].iloc[qp_idx[i]]):
                    #     print("lol")
                #print(df['post_content'].iloc[threads_idx[-1]:])

            # for j in threads_idx[:-1]:
            #     print(df['post_content'].iloc[threads_idx[j]:threads_idx[j+1]])
            # print(df['post_content'].iloc[threads_idx[-1]:])


        # increment


        #new_column = pd.DataFrame(({'new_header': ['new_value_1', 'new_value_2', 'new_value_3']}))
        #df['post_depth'] =

if __name__ == '__main__':
    CSVOperations().compute_depth()
