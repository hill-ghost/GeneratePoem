#-*- coding: UTF-8 -*-
import os
import re
import random

path = r'C:\Users\Pan\Desktop\poem\shixuehanying.txt'

def getCategory_Topics_Keywords():
	category = []
	topics = []
	keywords = []
	last_id = 0
	
	with open(path, 'r', encoding = 'UTF-8') as f:
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

if __name__ == '__main__':
	getCategory_Topics_Keywords()