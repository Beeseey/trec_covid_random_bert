import csv
import os
import json

def read_docid_texts(txt_file):

	file = open(txt_file, 'r', encoding='ISO-8859-1')
	text = file.read()
	file.close()

	dataset = list()
	for line in text.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		identifier = line  #.split('.')[0]
		if ';' not in identifier:
			dataset.append(identifier)
	
	return dataset

def read_metadata(csv_file_path):

	data_dict = dict()
	with open(csv_file_path, newline='', encoding="utf8") as csvfile:
			reader = csv.reader(csvfile)
			headers = next(reader, None)
			#reader = tqdm(reader)
			#reader.set_description("Creating Sequences")
			for row in reader:

				data_object = dict()
				data_object['cord_id'] = row[0]
				data_object['sha'] = row[1]
				data_object['source_x'] = row[2]
				data_object['title'] = row[3]
				data_object['doi'] = row[4]
				data_object['pmcid'] = row[5]
				data_object['pubmed_id'] = row[6]
				data_object['license'] = row[7]
				data_object['abstract'] = row[8]
				data_object['publish_time'] = row[9]
				data_object['authors'] = row[10]
				data_object['journal'] = row[11]
				#data_object['who_covidence'] = row[13]
				data_object['has_pdf'] = row[15]
				data_object['has_pmc'] = row[16]
				#data_object['full_text_file'] = row[16]
				data_object['url'] = row[17]
				data_dict[row[0]] = data_object
				
	return data_dict

def read_path_dicts(paths):

	path_dict = dict()
	for path in paths:
		filename = os.path.split(path)[1].split('.')[0]
		path_dict[filename] = path

	return path_dict

def read_text_from_json(path,pmc_or_pdf):
	
	file_dict = json.load(open(path, 'rb'))

	if pmc_or_pdf == 'pmc':
		
		#title = file_dict['metadata']['title']
		sections = []
		texts = []
		for text_list in file_dict['body_text']:
			sections.append(text_list['section'])
			texts.append(text_list['text'])
		
		sections = '\n'.join(sections)
		body = '\n'.join(texts)
		
		ref_texts = ''
		if len(file_dict['ref_entries'])>1:
			ref_texts = []
			for key in file_dict['ref_entries']:
				ref_texts.append(file_dict['ref_entries'][key]['text'])
			ref_texts = '\n'.join(ref_texts)

	elif pmc_or_pdf == 'pdf':
		#title = file_dict['metadata']['title']
		sections = []
		texts = []
		for text_list in file_dict['body_text']:
			sections.append(text_list['section'])
			texts.append(text_list['text'])
		sections = '\n'.join(sections)
		body = '\n'.join(texts)

		ref_texts = ''
		if len(file_dict['ref_entries'])>1:
			ref_texts = []
			for key in file_dict['ref_entries']:
				ref_texts.append(file_dict['ref_entries'][key]['text'])
			ref_texts = '\n'.join(ref_texts)

	return sections,body,ref_texts

		