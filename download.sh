#!/bin/bash

# donwload imdb
mkdir -p imdbface
cd imdbface

if [ ! -f imdb_crop.tar ]; then
    wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar
fi

if [ ! -d imdb_crop ]; then
    tar xf imdb_crop.tar
fi

if [ ! -f wiki_crop.tar ]; then
    wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar
fi

if [ ! -d wiki_crop ]; then
    tar xf wiki_crop.tar
fi

# donwload cifar

cd ../

if [ ! -f cifar-100-python.tar.gz ]; then
    wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
fi

if [ ! -d cifar-100-python ]; then
    tar -zxvf cifar-100-python.tar.gz
fi
