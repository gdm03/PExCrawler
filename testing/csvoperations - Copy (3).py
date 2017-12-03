import numpy as np
import pandas as pd

class CSVOperations():
    def compute_depth(self):
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
                current_string == new_df['quoted_post'].loc[idx]

                if ',<sep>,' in new_df['quoted_post'].loc[idx]:
                    strings = new_df['quoted_post'].loc[idx].split(',<sep>,')
                    depth_list = []

                match = new_df.index[new_df['post_content'] == [current_string for current_string in strings]].tolist()

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
    CSVOperations().compute_depth()
