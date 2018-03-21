# -*- coding: UTF-8 -*-
import os
import re
import random
import sys

non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)

path_shiXueHanYing = r'C:\Users\Pan\Desktop\poem\shixuehanying.txt'
path_pingShuiYun = r'C:\Users\Pan\Desktop\poem\pingshuiyun.txt'

PAT_CONTEXT = [
            [[2,0,0,1,1],[2,1,1,0,0],[2,1,0,0,1],[0,0,1,1,0]],
            [[0,0,1,1,0],[2,1,1,0,0],[2,1,2,0,1],[0,0,1,1,0]],
            [[2,1,0,0,1],[0,0,1,1,0],[2,0,0,1,1],[2,1,1,0,0]],
            [[2,1,1,0,0],[0,0,1,1,0],[2,0,0,1,1],[2,1,1,0,0]],
            [[2,0,2,1,0,0,1],[2,1,0,0,1,1,0],[2,1,2,0,0,1,1],[2,0,2,1,1,0,0]],
            [[2,0,2,1,1,0,0],[2,1,0,0,1,1,0],[2,1,2,0,0,1,1],[2,0,2,1,1,0,0]],
            [[2,1,2,0,0,1,1],[2,0,2,1,1,0,0],[2,0,2,1,0,0,1],[2,1,0,0,1,1,0]],
            [[2,1,0,0,1,1,0],[2,0,2,1,1,0,0],[2,0,2,1,0,0,1],[2,1,0,0,1,1,0]]]

def getCategory_Topics_Keywords():
    category = []
    topics = []
    keywords = []
    last_id = 0
    
    with open(path_shiXueHanYing, 'r', encoding = 'UTF-8') as f:
        temp_topics = []
        temp_keywords = []
        for line in f:
            line = line.strip(r'\n')
            temp = re.split('[\s]',line)[:-1]
            if temp[0] == '<begin>':
                category.append(temp[2])
            elif temp[0].isdigit():
                if last_id <= int(temp[0]):
                    temp_topics.append(temp[1])
                    temp_keywords.append(temp[2:])
                else:
                    topics.append(temp_topics)
                    temp_topics = []
                    temp_topics.append(temp[1])
                    keywords.append(temp_keywords)
                    temp_keywords = []
                    temp_keywords.append(temp[2:])
                last_id = int(temp[0])
        topics.append(temp_topics)
        keywords.append(temp_keywords)

    return category,topics,keywords

def random_keywords(list):
    length = len(list)
    index = random.randint(0,length-1)
    return list[index]

def read_pingShuiYun():
    with open(path_pingShuiYun, 'r', encoding = 'UTF-8') as f:
        titles = []
        context = []
        for line in f:
            line = line.strip(r'\n')
            temp = re.split('[\s]',line)
            titles.append(temp[0])
            context.append(temp[1])
    return titles,context

#获取韵部索引，-1为不押韵
def get_rhythm(titles,context,poem):
    if len(poem[0]) == 5:
        for i in range(0,106):
            if poem[1][4] in context[i] and poem[3][4] in context[i]:
                return i
            else:
                pass
    else:
        for i in range(0,106):
            if poem[1][6] in context[i] and poem[3][6] in context[i]:
                return i
            else:
                pass
    return -1

def pattern_match(poem_pattern):
    if len(poem_pattern[0]) == 5:
        for i in range(0,4):
            jump = False
            for j in range(0,4):
                for k in range(0,5):
                    if poem_pattern[j][k] == 2 or PAT_CONTEXT[i][j][k] == 2 or poem_pattern[j][k] == PAT_CONTEXT[i][j][k]:
                        if j == 3 and k == 4:
                            return i
                    else:
                        jump = True
                        break
                if jump:
                    break
    else:
        for i in range(4,8):
            jump = False
            for j in range(0,4):
                for k in range(0,7):
                    if poem_pattern[j][k] == 2 or PAT_CONTEXT[i][j][k] == 2 or poem_pattern[j][k] == PAT_CONTEXT[i][j][k]:
                        if j == 3 and k == 6:
                            return i
                    else:
                        jump = True
                        break
                if jump:
                    break
    return -1

# 获取一首诗的平仄模式，以数字表示0平1仄2通-1无
def get_tone(title,context,poem):
    poem_pattern = []
    length = len(poem[0])
    for i in range(0,4):
        temp_result = []
        for j in range(0,length):
            sum = 0
            time = 0
            for k in range(0,106):
                if poem[i][j] in context[k]:
                    if '平' in title[k] or '上' in title[k]:
                        pass
                    else:
                        sum = sum + 1
                    time = time + 1
            if sum != time:
                if sum == 0:
                    temp_result.append(0)
                else:
                    temp_result.append(2)
            else:
                if time == 0:
                    temp_result.append(-1)
                else:
                    temp_result.append(1)
        poem_pattern.append(temp_result)
    return poem_pattern

# 将以数字表示的平仄模式转化成汉字表示
def change_chinese(pattern):
    length = len(pattern[0])
    result = ''
    for i in range(0,4):
        for j in range(0,length):
            if pattern[i][j] == 0:
                result = result + '平'
            elif pattern[i][j] == 1:
                result = result + '仄'
            elif pattern[i][j] == 2:
                result = result + '通'
            else:
                result = result + '无' 
        result = result + ' '
    return result

if __name__ == '__main__':
    # getCategory_Topics_Keywords()
    # read_pingShuiYun()
    # print(change_chinese(PAT_CONTEXT[0]))
    title,context = read_pingShuiYun()
    test1 = [['四','时','运','灰','琯'],['一','夕','变','冬','春'],['送','寒','余','雪','尽'],['迎','岁','早','梅','新']]
    test2 = [['澄','潭','皎','镜','石','崔','巍'],['万','壑','千','岩','暗','绿','苔'],['林','亭','自','有','幽','贞','趣'],['况','复','秋','深','爽','气','来']]
    test3 = [['千','里','云','山','外'],['萧','萧','落','叶','时'],['乱','鸦','啼','雾','雨'],['残','月','挂','寒','枝']]
    print(change_chinese(get_tone(title,context,test3)))
    print(get_tone(title,context,test3))
    print(get_rhythm(title,context,test3))
    print(pattern_match(PAT_CONTEXT[1]))
    print(pattern_match(get_tone(title,context,test3)))
