import datetime

def string_to_delta(string_delta):
    if string_delta == "yesterday":
        return (datetime.datetime.now() - datetime.timedelta(1)).strftime("%b %d, %Y")

    value, unit, _ = string_delta.split()
    unit_list = ['hour', 'day', 'week']
    if unit in unit_list:
        unit += 's'
    return (datetime.datetime.now() - datetime.timedelta(**{unit: float(value)})).strftime("%b %d, %Y")

if __name__ == '__main__':
    testlist = ['1 days ago', 'yesterday', "Jun 15 2009"]
    [print(string_to_delta(s)) if "ago" in s or "yesterday" in s else print(s) for s in testlist]