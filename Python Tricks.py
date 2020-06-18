import timeit


def calc(args):
    if args.operation == 'add':
        return args.x + args.y
    elif args.operation == 'sub':
        return args.x - args.y
    elif args.operation == 'mul':
        return args.x * args.y
    elif args.operation == 'div':
        return args.x / args.y


import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, default=1.0,
                        help='What is the first number?')
    parser.add_argument('--y', type=float, default=1.0,
                        help='What is the second number?')
    parser.add_argument('--operation', type=str, default='add',
                        help='What operation? Can choose add, sub, mul, or div')
    args = parser.parse_args()
    sys.stdout.write(str(calc(args)))


def calc(args):
    if args.operation == 'add':
        return args.x + args.y
    elif args.operation == 'sub':
        return args.x - args.y
    elif args.operation == 'mul':
        return args.x * args.y
    elif args.operation == 'div':
        return args.x / args.y


if __name__ == '__main__':
    main()

'''
#python argparse_example.py --x=5 --y=3 --operation=mul

#python argparse_example.py -h

#usage: argparse_example.py [-h] [--x X] [--y Y] [--operation OPERATION]

# optional arguments:
#   -h, --help            show this help message and exit
#   --x X                 What is the first number?
#   --y Y                 What is the second number?
#   --operation OPERATION
#                         What operation? Can choose add, sub, mul, or div
'''


input_list = [5,6,2,1,6,7,10,12]

def div_by_five(num):
    if num % 5 == 0:
        return True
    else:
        return False

xyz = (i for i in input_list if div_by_five(i))
print(list(xyz))

xyz = [i for i in input_list if div_by_five(i)]
print(xyz)

[print(i) for i in range(5)]


[[print(i,ii) for ii in range(3)] for i in range(5)]


# Generators
CORRECT_COMBO = (3, 6, 1)

found_combo = False
for c1 in range(10):
    if found_combo:
        break
    for c2 in range(10):
        if found_combo:
            break
        for c3 in range(10):
            if (c1, c2, c3) == CORRECT_COMBO:
                print('Found the combo:{}'.format((c1, c2, c3)))
                found_combo = True
                break



def combo_gen():
    for c1 in range(10):
        for c2 in range(10):
            for c3 in range(10):
                yield (c1, c2, c3)

for (c1, c2, c3) in combo_gen():
    print(c1, c2, c3)
    if (c1, c2, c3) == CORRECT_COMBO:
        print('Found the combo:{}'.format((c1, c2, c3)))
        break


# Multiprocessing

import multiprocessing

def spawn(num, num2):
    print('Spawn # {} {}'.format(num, num2))

if __name__ == '__main__':
    for i in range(5):
        p = multiprocessing.Process(target=spawn, args=(i, i+1))
        p.start()



# Spider Example

from multiprocessing import Pool
import bs4 as bs
import random
import requests
import string

def random_starting_url():
    starting = ''.join(random.SystemRandom().choice(string.ascii_lowercase) for _ in range(3))
    url = ''.join(['http://', starting, '.com'])
    return url

def handle_local_links(url,link):
    if link.startswith('/'):
        return ''.join([url,link])
    else:
        return link

def get_links(url):
    resp = requests.get(url)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    body = soup.body
    links = [link.get('href') for link in body.find_all('a')]
    links = [handle_local_links(url,link) for link in links]
    links = [str(link.encode("ascii")) for link in links]
    return links

def get_links(url):
    try:
        resp = requests.get(url)
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        body = soup.body
        links = [link.get('href') for link in body.find_all('a')]
        links = [handle_local_links(url,link) for link in links]
        links = [str(link.encode("ascii")) for link in links]
        return links

    except TypeError as e:
        print(e)
        print('Got a TypeError, probably got a None that we tried to iterate over')
        return []
    except IndexError as e:
        print(e)
        print('We probably did not find any useful links, returning empty list')
        return []
    except AttributeError as e:
        print(e)
        print('Likely got None for links, so we are throwing this')
        return []
    except Exception as e:
        print(str(e))
        # log this error
        return []


def main():
    how_many = 50
    p = Pool(processes=how_many)
    parse_us = [random_starting_url() for _ in range(how_many)]

    data = p.map(get_links, [link for link in parse_us])
    data = [url for url_list in data for url in url_list]
    p.close()

    with open('urls.txt', 'w') as f:
        f.write(str(data))


if __name__ == '__main__':
    main()





import torch, time.timeit
t_cpu = torch.rand(500,500,500)
%timeit t_cpu @ t_cpu

t_gpu = torch.rand(500,500,500).cuda()
%timeit t_gpu @ t_gpu



from fastai import *
