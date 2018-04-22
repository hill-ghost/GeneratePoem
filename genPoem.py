#-*- coding: UTF-8 -*-
from testSentence import *
from testPoem import *
from tools import *

title,context = read_pingShuiYun()

complete = False
category,topics,keywords = getCategory_Topics_Keywords()

word_num = int(input('请输入5或7表示需要生成的诗句类型：'))
while word_num != 5 and word_num != 7:
    print('输入错误，请输入5或7表示需要生成的诗句类型：')
    word_num = int(input())
    
print('共有以下类别:')
num = 0
for name in category:
    print(name + '%s' % num)
    num = num + 1
row = input('请输入类别的代号:')
row = int(row)

print('该类别下共有以下主题:')
num = 0
for name in topics[row]:
    print(name + '%s' % num)
    num = num + 1
col = input('请输入主题的代号:')
col = int(col)

while not complete:
    first = gen_firstSentence(word_num,row,col)
    if len(first) > word_num:
        print('字数超过')
        continue
    if len(first) < word_num:
        print('字数不足')
        continue
    result = genNextSentence(first,word_num)
    print('平仄模式匹配分数为：%s' % pattern_match_score(get_tone(title,context,result)))
    print('韵部为：%s' % get_rhythm(title,context,result))
    if pattern_match_score(get_tone(title,context,result)) > 0.8 and get_rhythm(title,context,result) > -1:
        break
    else:
        print('不符合要求，重新生成。')

print(result)
        
