import pandas as pd
import re
import tablib


def convertJsonToXlsx(json_path, xlsx_path):
    js = pd.read_json(json_path)
    all_result = js.results['all']
    mean_result = js.results['mean']
    header = ['patient']
    for i in all_result[0]['1'].keys():
        header.append(i)
    data = []
    for result in all_result:
        name = re.sub(r'\D', "", result['reference'].split('/')[-1])
        body = [name]
        for v in result['1'].values():
            body.append(v)
        data.append(tuple(body))

    body = ['mean']
    for v in mean_result['1'].values():
        body.append(v)
    data.append(tuple(body))

    data = tablib.Dataset(*data, headers=header)
    open(xlsx_path, 'wb').write(data.xlsx)


if __name__ == '__main__':
    json_path = r'/media/lu/Data/Data/nnUnet/predict/bladder/190/ensemble/summary.json'
    xlsx_path = r'/media/lu/Data/Data/nnUnet/predict/bladder/190/ensemble/summary.xlsx'
    convertJsonToXlsx(json_path, xlsx_path)