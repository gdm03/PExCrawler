quoted_post = ["Go with a brand new car. There's the ",
                'silent',
                ' 3-year rule daw na naglalabasan na ang mga issues after 3 '
                'years lalo na kung super gamit yung sasakyan so check '
                'mileage, how many times na dinala sa casa, and so on.']

post_content = ["\nGo with a brand new car. There's the ",
                 'silent',
                 ' 3-year rule daw na naglalabasan na ang mga issues after 3 '
                 'years lalo na kung super gamit yung sasakyan so check '
                 'mileage, how many times na dinala sa casa, and so on.\n']

print(quoted_post)
print(post_content)
print("\n")
print([s.strip() for s in quoted_post])
print([s.strip() for s in post_content])
print(' '.join([s.strip() for s in quoted_post]))
print(' '.join([s.strip() for s in post_content]))
print("\n")