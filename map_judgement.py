import csv 
import xml.etree.ElementTree as ET

def read_topics(xml_file):

	tree = ET.parse(xml_file)
	root = tree.getroot()
	data_dict = {}
	for item in root.findall('topic'):
		query_id = item.attrib['number']
		for child in item:
			if child.tag == 'query':
				query = child.text
			elif child.tag == 'question':
				question = child.text
		data_dict[query_id] = query+' '+question
	return data_dict

def read_rel_judge(txt_file):

	file = open(txt_file, 'r', encoding='ISO-8859-1')
	text = file.read()
	file.close()

	data_dict = {}
	query_info_list = []
	_query = '1'
	for run in text.split('\n'):
		run_dict = {}
		if run != '':
			query,itr,_,doc_id,judg = run.split(' ')
			run_dict['rel'] = judg
			run_dict['id'] = doc_id
		else:
			query = '31'
		
		if query != _query:
			doc_idx_rel = []
			doc_idx_not_rel = []
			doc_idx_partial = []
			for info_dict in query_info_list:
				if info_dict['rel'] == '0':
					doc_idx_not_rel.append(info_dict['id'])
				elif info_dict['rel'] == '1':
					doc_idx_partial.append(info_dict['id'])
				elif info_dict['rel'] == '2':
					doc_idx_rel.append(info_dict['id'])

			data_dict[_query] = {'rel':doc_idx_rel,'not_rel':doc_idx_not_rel,'partial':doc_idx_partial}
			query_info_list = []
			
		query_info_list.append(run_dict)
		_query = query
		
	return data_dict

data_dict = read_rel_judge('qrels-rnd1.txt')

topic_dict = read_topics('topics-rnd1.xml')

#print(data_dict)

#raise('stop')


def read_result(txt_file):

	file = open(txt_file, 'r', encoding='ISO-8859-1')
	text = file.read()
	file.close()

	return text

result = read_result('alam-x.txt')
file = open('mapped_result.txt', 'w')
count_correct = 0
count = 0

for line in result.split('\n'):

	if line == '':
		break

	line = line.replace("'",'').strip('][').split(',')
	query = line[0]

	jdg_dict = data_dict[query]

	line[2] = line[2].replace(' ','')

	if line[2] in jdg_dict['rel']:
		count_correct+=1
		jdg = 2
	elif line[2] in jdg_dict['not_rel']:
		count+=1
		jdg = 0
	elif line[2] in jdg_dict['partial']:
		count_correct+=1
		jdg = 1
	else:
		jdg = 0
	count+=1

	text = query+line[1]+' '+line[2]+line[3]+line[4]+' '+str(jdg)+'\n'
	file.write(text)

file.close()

print(count,'documents judged',count_correct,'correctly retrieved')