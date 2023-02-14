# def write_raw_index(file):
#     filename = file
#     with open(filename, 'r+', encoding='utf-8') as f:
#         content = f.read()
#         f.seek(0, 0)
#         #mid, text, source, uid
#         text = 'data' + ',' + 'value'
#         f.write(text + '\n' + content)
#
# write_raw_index('E:\\mit\\masterarbeit\\meta learning\\data analyse\\small csv\\new\\d12 normal1-1.csv')

import os


def get_files(path='E:\\mit\\masterarbeit\\meta learning\\data analyse\\small csv\\new', rule=".csv"):
    for fpathe, dirs, fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe, f)
            if filename.endswith(rule):
                with open(filename, 'r+', encoding='UTF-8-sig') as f:
                    content = f.read()
                    f.seek(0, 0)
                    # mid, text, source, uid
                    text = 'date' + ',' + 'value'
                    f.write(text + '\n' + content)


get_files()
