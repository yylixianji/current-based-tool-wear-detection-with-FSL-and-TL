import json

current = {"class_names": ["N", "L"],
           "class_roots": ["E:\\mit\\masterarbeit\\easy-few-shot-learning-master\\data\\current\\N",
                           "E:\\mit\\masterarbeit\\easy-few-shot-learning-master\\data\\current\\L"]}
file = open('current.txt', 'w', encoding='utf8')
json.dump(current, file)
file.close()
