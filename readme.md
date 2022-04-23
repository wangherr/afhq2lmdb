## A script to transform afhq dataset to lmdb format

-----------------------
original afhq directory level
~~~
|-afhg
    |-train
        |-cat
            |-flickr_cat_000002.jpg
            ...
        |-dog
        |-wild
    |-val
        |-cat
        |-dog
        |-wild
~~~

lmdb directory level
~~~
|-afhg_lmdb
    |-cat
        |-cat
            |-train.lmdb
                |-data.mdb
                |-lock.mdb
            |-val.lmdb
        |-dog
        |-wild
    |-dog
    |-wild
~~~


## use

1. download and unzip afhq
2. run the follow command：
~~~
python main.py --input_root 'afhq' --out_root 'afhq_lmdb'
~~~

## description

In the lmdb, the image's shape is (521, 512, 3)