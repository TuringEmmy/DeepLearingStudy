

## 它已经不仅仅能够作古诗，还能模仿周杰伦创作歌词！！



```
我的你的她
蛾眉脚的泪花
乱飞从慌乱
笛卡尔的悲伤
迟早在是石板上
荒废了晚上
夜你的她不是她
....
```




## 阅遍了近4万首唐诗

```
龙舆迎池里，控列守龙猱。
几岁芳篁落，来和晚月中。
殊乘暮心处，麦光属激羁。
铁门通眼峡，高桂露沙连。
倘子门中望，何妨嶮锦楼。
择闻洛臣识，椒苑根觞吼。
柳翰天河酒，光方入胶明。
```






* 使用方法：
```
# for poem train
python3 main.py -w poem --train
# for lyric train
python3 main.py -w lyric --train

# for generate poem
python3 main.py -w poem --no-train
# for generate lyric
python3 main.py -w lyric --no-train

```

* 参数说明
`-w or --write`: 设置作诗还是创作歌词，poem表示诗，lyric表示歌词
`--train`: 训练标识位，首次运行请先train一下...
`--no-train`: 生成标识位

训练的时候有点慢，有GPU就更好啦，最后gen的时候你就可以看到我们牛逼掉渣天的诗啦！

这是它做的诗：

```
龙舆迎池里，控列守龙猱。
几岁芳篁落，来和晚月中。
殊乘暮心处，麦光属激羁。
铁门通眼峡，高桂露沙连。
倘子门中望，何妨嶮锦楼。
择闻洛臣识，椒苑根觞吼。
柳翰天河酒，光方入胶明
```
感觉有一种李白的豪放风度！

这是它作的歌词：

 ```
 我的你的她
 蛾眉脚的泪花
 乱飞从慌乱
 笛卡尔的悲伤
 迟早在是石板上
 荒废了晚上
 夜你的她不是她
 ....
 ```


### Author & Cite
This repo implement by Jin Fagang.
(c) Jin Fagang.
Blog: [jinfagang.github.io](https://jinfagang.github.io)
