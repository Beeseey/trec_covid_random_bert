import csv 
import xml.etree.ElementTree as ET
from bert.tokenization import bert_tokenization
from utils.parse_utils import read_docid_texts, read_metadata, read_path_dicts, read_text_from_json
from glob import glob
import os
import csv
from tqdm import tqdm
import gc


TOKENIZER = createTokenizer()

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

def read_rel_(txt_file):

	file = open(txt_file, 'r', encoding='ISO-8859-1')
	text = file.read()
	file.close()

	doc_ids = []

	for run in text.split('\n'):
		run_dict = {}
		if run != '':
			query,itr,_,doc_id,judg = run.split(' ')
			doc_ids.append(doc_id)

	return doc_ids

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

			data_dict[_query] = {'rel':doc_idx_rel[:],'not_rel':doc_idx_not_rel[:],'partial':doc_idx_partial[:]}
			query_info_list = []
			
		query_info_list.append(run_dict)
		_query = query
		
	return data_dict
def read_random(txt_file):
	file = open(txt_file, 'r', encoding='ISO-8859-1')
	text = file.read()
	file.close()

	_query = '35'
	data_dict = {}
	query_info_list = []

	rel_doc_list = read_rel_('qrels-rnd1.txt')

	for run in text.split('\n'):
		
		run_dict = {}
		if run != '':
			line = run.replace("'",'').strip('][').split(',')
			query = line[0].replace(' ','')
			doc_id = line[2].replace(' ','')
			run_dict['id'] = doc_id
		else:
			query = '36'

		if query != _query and query == '36':
			doc_idx_rel = []
			print(len(query_info_list))
			for info_dict in query_info_list:
				doc_idx_rel.append(info_dict['id'])
			data_dict[_query] = {'rel':doc_idx_rel}
			query_info_list = []
		if doc_id not in rel_doc_list and query == '35':
			query_info_list.append(run_dict)
		_query = query

	return data_dict

def read_corpus(docids,metadata_dict,path_dict):

	data_dict = {}

	for idx in docids:
		data_object_dict = metadata_dict[idx]
		abstract = data_object_dict['abstract']
		title = data_object_dict['title']
		full_text = '','',''
		if data_object_dict['has_pdf'] == 'True':
			if ' ' in data_object_dict['sha']:
				data_object_dict['sha'] = data_object_dict['sha'].split(' ')[0]		
				data_object_dict['sha'] = data_object_dict['sha'].replace(';','')
			json_file_path = path_dict[data_object_dict['sha']]
			full_text = read_text_from_json(json_file_path,'pdf')
		elif data_object_dict['has_pmc'] == 'True':
			try:
				json_file_path = path_dict[data_object_dict['pmcid']]
				full_text = read_text_from_json(json_file_path,'pmc')
				json_dicts.append(data_object_dict)
			except:
				pass
		else:
			pass
		sections,body,ref_texts = full_text
		f_t = abstract + ' '+ title +' '+ sections+' '+body+' '+ref_texts
		data_dict[idx] = {'abstract':abstract,'title':title,'fulltext':f_t}
	
	return(data_dict)

def createTokenizer():
	currentDir = os.path.dirname(os.path.realpath(__file__))
	modelsFolder = os.path.join(currentDir, "models", "multi_cased_L-12_H-768_A-12")
	vocab_file = os.path.join(modelsFolder, "vocab.txt")

	tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
	return tokenizer

def get_data():
	global TOKENIZER

	metadata_dict = read_metadata('metadata.csv')

	biorxiv_medrxiv = [path for path in glob(os.path.join('C:/Users/mdrah/NIST_COVID/biorxiv_medrxiv/biorxiv_medrxiv','*','*'))]
	comm_use_subset = [path for path in glob(os.path.join('C:/Users/mdrah/NIST_COVID/comm_use_subset/comm_use_subset','*','*'))]
	custom_license = [path for path in glob(os.path.join('C:/Users/mdrah/NIST_COVID/custom_license/custom_license','*','*'))]
	noncomm_use_subset = [path for path in glob(os.path.join('C:/Users/mdrah/NIST_COVID/noncomm_use_subset/noncomm_use_subset','*','*'))]

	complete_paths = biorxiv_medrxiv+custom_license+comm_use_subset+noncomm_use_subset

	path_dict = read_path_dicts(complete_paths)

	rel_doc_dict = read_rel_judge('qrels-rnd1.txt')

	topic_dict = read_topics('topics-rnd1.xml')

	docids = read_docid_texts('docids-rnd1.txt')

	corpus_dict = read_corpus(docids,metadata_dict,path_dict)

	X,X1,Y = [],[],[]
	data = []

	count = 0

	for query in rel_doc_dict:
		
		for judg in rel_doc_dict[query]:
			count2=0

			for cord_id in rel_doc_dict[query][judg]:
				if cord_id in docids:
					topic = topic_dict[query]
					doc = corpus_dict[cord_id]
					if judg == 'rel':
						class_ = 2
					elif judg == 'partial':
						class_ = 1
					elif judg == 'not_rel':
						class_ = 0
					
					full_text = doc['fulltext']
					#abstract = doc['abstract']
					#title_abstract = title+' '+abstract
					full_text = full_text.split(' ')
					num_of_divs = len(full_text)//300
					topic_token = tokenizer.tokenize(topic)
					
					for i in range(num_of_divs):
						start = i * 300
						end = start+300
						txt = full_text[start:end]
						txt = ' '.join(txt)
						txt_token = tokenizer.tokenize(txt)		
						if len(txt_token)+len(topic_token) <= 509:
							gc.disable()
							data.append([topic_token,txt_token,class_])
			
						
#					
#					if len(topic.split(' '))+len(title_abstract.split(' ')) > 200:
#						diff = 200 - len(topic.split(' '))
#						title_abstract = title_abstract.split(' ')
#						title_abstract = title_abstract[:diff]
#						title_abstract = ' '.join(title_abstract)
#
#					X1.append(title_abstract)
#					Y.append(class_)
#					'''
		count+=1
		print(count,'out of',len(rel_doc_dict))
	#return X,X1,Y
	return data

def get_data_():
	metadata_dict = read_metadata('corpus2/metadata.csv')

	arxiv = [path for path in glob(os.path.join('C:/Users/mdrah/NIST_COVID/corpus2/arxiv','*','*'))]
	biorxiv_medrxiv = [path for path in glob(os.path.join('C:/Users/mdrah/NIST_COVID/corpus2/biorxiv_medrxiv','*','*'))]
	comm_use_subset = [path for path in glob(os.path.join('C:/Users/mdrah/NIST_COVID/corpus2/comm_use_subset','*','*'))]
	custom_license = [path for path in glob(os.path.join('C:/Users/mdrah/NIST_COVID/corpus2/custom_license','*','*'))]
	noncomm_use_subset = [path for path in glob(os.path.join('C:/Users/mdrah/NIST_COVID/corpus2/noncomm_use_subset','*','*'))]

	complete_paths = biorxiv_medrxiv+custom_license+comm_use_subset+noncomm_use_subset+arxiv

	path_dict = read_path_dicts(complete_paths)

	rel_doc_dict = read_random('random_bert_4k.txt')
	topic_dict = read_topics('topics-rnd2.xml')
	docids = read_docid_texts('corpus2/docids-rnd2.txt')
	corpus_dict = read_corpus(docids,metadata_dict,path_dict)
	print(corpus_dict['ivfvu5i3'])

	X,X1 = [],[]
	count = 0

	for query in rel_doc_dict:
		for cord_id in rel_doc_dict[query]['rel']:
			x_ = {}
			if cord_id in docids:
				topic = topic_dict[query]
				doc = corpus_dict[cord_id]
				X.append(topic)
				title = doc['title']
				abstract = doc['abstract']
				title_abstract = title+' '+abstract
				if len(topic.split(' '))+len(title_abstract.split(' ')) > 200:
					diff = 200 - len(topic.split(' '))
					title_abstract = title_abstract.split(' ')
					title_abstract = title_abstract[:diff]
					title_abstract = ' '.join(title_abstract)
				x_ = {'X':topic,'X1':title_abstract,'query':query,'cord_id':cord_id, 'title': title}
				X1.append(x_)

		count+=1
		print(len(X1))
		print(count,'out of',len(rel_doc_dict))
	return X1

'''

def read_result(txt_file):

	file = open(txt_file, 'r', encoding='ISO-8859-1')
	text = file.read()
	file.close()

	return text

result = read_result('alam-x.txt')
#file = open('mapped_result.txt', 'w')
count = 0

for line in result.split('\n'):

	if line == '':
		break

	line = line.replace("'",'').strip('][').split(',')
	query = line[0]

	jdg_dict = data_dict[query]

	if line[2] in jdg_dict['rel']:
		jdg = 2
	elif line[2] in jdg_dict['not_rel']:
		jdg = 0
	elif line[2] in jdg_dict['partial']:
		jdg = 1
	else:
		jdg = 0

	text = query+line[1]+line[2]+line[3]+line[4]+' '+str(jdg)+'\n'
	break
	#file.write(text)

#file.close()
'''