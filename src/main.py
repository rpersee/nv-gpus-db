#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from functools import partial, reduce
import json
from datetime import datetime


def css_escape(string: str) -> str:
    return re.sub(r'([\[\]{}()])', r'\\\1', string)


def compose(*fns):
    return reduce(lambda f, g: lambda x: f(g(x)), fns, lambda x: x)


def clean(*strs: str) -> str:
    # bottom-up function execution
    cleaner = compose(
        str.strip,  # remove leading and trailling spaces
        partial(re.sub, r'\[.*?\]', ''),  # remove references like '[1]'
    )
    for s in strs:
        yield cleaner(s)


def fmt_headers(*strs) -> str:
    # remove duplicates while while preserving order
    for s in dict.fromkeys(strs).keys():
        yield s


def find_flat_children(tag, name, **attrs):
    for sibling in tag.next_siblings:
        if sibling.name == tag.name:
            return
        if sibling.name != name:
            continue
        if not all(sibling.attrs[key] == value for key, value in attrs.items()):
            continue
        yield sibling


if __name__ == "__main__":
    r = requests.get(
        'https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units')
    soup = BeautifulSoup(r.text, 'lxml')
    date_now = datetime.now()

    dataframes = {}

    for h2 in soup.select('#mw-content-text > div.mw-parser-output > h2'):
        for h3 in find_flat_children(h2, 'h3'):
            table = next(find_flat_children(h3, 'table'), None)
            if not table:
                continue

            df = pd.concat(pd.read_html(str(table)), axis=1)
            cat_name = h2.select_one('span.mw-headline').text
            gen_name = h3.select_one('span.mw-headline').text

            if not cat_name in dataframes:
                dataframes[cat_name] = dict()
            dataframes[cat_name][gen_name] = df

    parsed = {
        'date': date_now.isoformat(),
        'categories': []
    }

    for cat_name in dataframes.keys():
        category = {}
        category['name'] = cat_name
        category['generations'] = []

        for gen_name in dataframes[cat_name].keys():
            generation = {}
            generation['name'] = gen_name

            cleaned = dataframes[cat_name][gen_name].applymap(
                lambda s: next(clean(s)) if isinstance(s, str) else s)
            cleaned.columns = [': '.join(fmt_headers(*clean(*col)))
                               for col in cleaned.columns]

            generation['gpus'] = cleaned.to_dict(orient='records')
            category['generations'].append(generation)
        parsed['categories'].append(category)

    print(json.dumps(parsed, indent=4))
