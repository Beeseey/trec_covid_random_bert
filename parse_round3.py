from utils.parse_utils import read_docid_texts, read_metadata, read_path_dicts, read_text_from_json
from tqdm import tqdm
from glob import glob
import os
import csv
import pickle
#pip install tqdm
from tqdm import tqdm

list_of_docids = read_docid_texts('../covid3/docids-rnd3.txt')

metadata_dict = read_metadata('../covid3/metadata_3.csv')

metadata_dict2 = read_metadata('../corpus2/metadata.csv')

arxiv = [path for path in glob(os.path.join('C:/Users/mdrah/NIST_COVID/corpus2/arxiv','*','*'))]
comm_use_subset = [path for path in glob(os.path.join('C:/Users/mdrah/NIST_COVID/corpus2/comm_use_subset','*','*'))]
custom_license = [path for path in glob(os.path.join('C:/Users/mdrah/NIST_COVID/corpus2/custom_license','*','*'))]
noncomm_use_subset = [path for path in glob(os.path.join('C:/Users/mdrah/NIST_COVID/corpus2/noncomm_use_subset','*','*'))]
biorxiv_medrxiv = [path for path in glob(os.path.join('C:/Users/mdrah/NIST_COVID/corpus2/biorxiv_medrxiv','*','*'))]
complete_paths = biorxiv_medrxiv+custom_license+comm_use_subset+noncomm_use_subset+arxiv

path_dict = read_path_dicts(complete_paths)

unread = list()
json_dicts = list()

root_Folder = '..'

'''
file = open("corpus1.pkl","wb")
file2 = open("corpus2.pkl","wb")
file3 = open("corpus3.pkl","wb")
'''
file = open('../covid3/corpus3.txt', 'w+',encoding='utf-8')
csv_file = open('../covid3/corpus3.csv', 'w+', newline='',encoding='utf-8')
csv_writer = csv.DictWriter(csv_file,fieldnames=['docid','abstract','title','sections','body','ref_texts'])
csv_writer.writeheader()

list_of_docids = tqdm(list_of_docids)
list_of_docids.set_description("Parsing data")

#text_ = []

for idx in list_of_docids:

	parsed_data = dict()
	corpus2 = False
	try:
		data_object_dict = metadata_dict[idx]
	except:
		if idx in list(metadata_dict2.keys()):
			data_object_dict = metadata_dict2[idx]
			corpus2 = True
		else:
			corpus2 = 'skip'

	#print(data_object_dict)

	abstract = data_object_dict['abstract']
	title = data_object_dict['title']
	sections,body,ref_texts = '','',''

	if not corpus2: 
		if data_object_dict['has_pdf'] != '':
		
			#json_file_path = path_dict[data_object_dict['sha']]
			if ';' in data_object_dict['has_pdf']:
				data_object_dict['has_pdf'] = data_object_dict['has_pdf'].split('; ')[1]
				#print(data_object_dict['has_pdf'])	
			path = os.path.join(root_Folder,data_object_dict['has_pdf'])
			#print(path)
			full_text = read_text_from_json(path,'pdf')
			json_dicts.append(data_object_dict)

		elif data_object_dict['has_pmc'] != '':

			try:
				#json_file_path = path_dict[data_object_dict['pmcid']]
				path = os.path.join(root_Folder,data_object_dict['has_pmc'])
				full_text = read_text_from_json(path,'pmc')
				json_dicts.append(data_object_dict)
			except:
				continue

		else:
			unread.append(data_object_dict)
	elif corpus2 == True:

		if data_object_dict['has_pdf'] == 'True':

			if ' ' in data_object_dict['sha']:
				data_object_dict['sha'] = data_object_dict['sha'].split(' ')[0]		
				data_object_dict['sha'] = data_object_dict['sha'].replace(';','')

			json_file_path = path_dict[data_object_dict['sha']]
			full_text = read_text_from_json(json_file_path,'pdf')
			json_dicts.append(data_object_dict)

		elif data_object_dict['has_pmc'] == 'True':

			try:

				json_file_path = path_dict[data_object_dict['pmcid']]
				full_text = read_text_from_json(json_file_path,'pmc')
				json_dicts.append(data_object_dict)
			except:
				continue

		else:
			unread.append(data_object_dict)
	else:
		pass

	sections,body,ref_texts = full_text

	text = '\n'.join([abstract]+[title]+list(full_text))

	text = '<P ID='+idx+'>'+'\n'+text+'\n'+'</P>'

	#text_.append(text)
	
	parsed_data['docid'] = idx
	parsed_data['abstract'] = abstract.replace('\n',' ')
	parsed_data['title'] = title.replace('\n','')
	parsed_data['sections'] = sections.replace('\n',' ')
	parsed_data['body'] = body.replace('\n','')
	parsed_data['ref_texts'] = ref_texts.replace('\n',' ')

	file.write(text+"\n")
	csv_writer.writerow(parsed_data)
'''
a = text_[:25000]
b = text_[25000:50000]
c = text_[50000:]
#text = ' '.join(a)
pickle.dump(a,file)
pickle.dump(b,file2)
pickle.dump(c,file3)
#text = ' '.join(b)
#pickle.dump(text,file2)
#print(len(json_dicts))
#print(len(unread))
'''
file.close()
csv_file.close()