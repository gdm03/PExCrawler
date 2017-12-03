import datetime

class TimeConv():
    #string_delta = '1 hour ago'

    # string_delta = '2 hours ago'
    # string_delta = '1 day ago'
    # string_delta = '2 days ago'
    # string_delta = '1 week ago'
    # string_delta = '2 weeks ago'
    end = datetime.datetime(2017, 10, 26, 3, 1, 18, 862859)
    start = datetime.datetime(2017, 10, 26, 2, 28, 2, 339696)
    res = end - start
    print(start, end)

    print(res)
    print(divmod(res.days * 86400 + res.seconds, 60))
    def string_to_delta(self, string_delta):
        value, unit, _ = string_delta.split()
        unit_list = ['hour', 'day', 'week']
        if unit in unit_list:
            unit += 's'

        return (datetime.datetime.now() - datetime.timedelta(**{unit: float(value)})).strftime("%d-%b-%y")

    #print(string_to_delta('1 hour ago'))

tc = TimeConv()
print(tc.string_to_delta('24 hour ago'))