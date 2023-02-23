#!/usr/bin/env python3

import json
import re
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from functools import partial, reduce

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag


def css_escape(string: str) -> str:
    """Escapes special characters in a CSS selector string.

    :param string: The CSS selector string to escape.
    :return: The escaped CSS selector string.
    """
    return re.sub(r'([\[\]{}()])', r'\\\1', string)


def compose(*fns: Callable) -> Callable:
    """Composes multiple functions into a single function.

    :param fns: The functions to compose.
    :return: The composed function.
    """

    return reduce(lambda f, g: lambda x: f(g(x)), fns, lambda x: x)


# noinspection PyDictDuplicateKeys
unicode_trans = str.maketrans({
    '\N{NO-BREAK SPACE}': ' ',  # \u00a0
    '\N{SOFT HYPHEN}': '',  # \u00ad
    '\N{NON-BREAKING HYPHEN}': '-',  # \u2011
    '\N{FIGURE DASH}': '-',  # \u2012
    '\N{EN DASH}': '-',  # \u2013
    '\N{EM DASH}': '-',  # \u2014
    '\N{MULTIPLICATION SIGN}': 'x',  # \u00D7
})

# bottom-up function execution
clean = compose(
    str.strip,  # remove leading and trailing spaces
    lambda s: s.translate(unicode_trans),  # replace unicode chars
    partial(re.sub, r'\[.*?\]', ''),  # remove references like '[1]'
)


def uniq(*strs: str) -> Iterator[str]:
    """Filter out duplicated strings.

    :param strs: The strings to deduplicate.
    :return: An iterator over the deduplicated strings.
    """

    # remove duplicates while preserving order
    yield from dict.fromkeys(strs).keys()


# noinspection PyUnresolvedReferences
def find_flat_children(tag: Tag, name: str, **attrs: dict[str, str]) -> Iterator[Tag]:
    """Returns an iterator over sibling HTML tags with the specified tag name and attributes,
    until a tag with the same name is found.

    :param tag: The HTML tag whose siblings to search.
    :param name: The tag name to search for.
    :param attrs: A dictionary of attributes and their values that the tags must have.
    :return: An iterator over the matching HTML tags.
    """
    for sibling in tag.next_siblings:
        if sibling.name == tag.name:
            return
        if sibling.name != name:
            continue
        if not all(sibling.attrs[key] == value for key, value in attrs.items()):
            continue
        yield sibling


if __name__ == "__main__":
    r = requests.get('https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units')
    soup = BeautifulSoup(r.text, 'lxml')
    # set time zone and remove microseconds to conform to the JSON Schema ISO8601 date-time format
    date_now = datetime.now(timezone.utc).replace(microsecond=0)

    dataframes = {}

    for h2 in soup.select('#mw-content-text > div.mw-parser-output > h2'):
        for h3 in find_flat_children(h2, 'h3'):
            table = next(find_flat_children(h3, 'table'), None)
            if table is None:
                continue

            # concatenate all tables found in the current h3 element
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
        category = {
            'name': cat_name,
            'generations': []
        }

        for gen_name in dataframes[cat_name].keys():
            generation = {
                'name': gen_name
            }

            df = dataframes[cat_name][gen_name]

            # apply clean() on string columns
            str_cols = df.columns[df.dtypes == 'object']
            df[str_cols] = df[str_cols].fillna("NaN").applymap(clean)

            # apply clean() and join multi-row headers
            df.columns = df.columns.map(lambda col: ': '.join(uniq(*map(clean, col))))

            generation['gpus'] = df.to_dict(orient='records')
            category['generations'].append(generation)

        parsed['categories'].append(category)

    print(json.dumps(parsed, indent=4))
