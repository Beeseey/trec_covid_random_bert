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

unread = list()
json_dicts = list()

root_Folder = '..'

file = open('../covid3/corpus3.txt', 'w+',encoding='utf-8')
csv_file = open('../covid3/corpus3.csv', 'w+', newline='',encoding='utf-8')
csv_writer = csv.DictWriter(csv_file,fieldnames=['docid','abstract','title','sections','body','ref_texts'])
csv_writer.writeheader()

list_of_docids = tqdm(list_of_docids)
list_of_docids.set_description("Parsing data")

for idx in list_of_docids:

	parsed_data = dict()
	inMetadataFile = True
	try:
		data_object_dict = metadata_dict[idx]
	except:
		inMetadataFile = False


	if inMetadataFile:
		abstract = data_object_dict['abstract']
		title = data_object_dict['title']
		sections,body,ref_texts = '','',''
		
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
	else:
		unread.append(data_object_dict)

print(len(unread),'unread files')

file.close()
csv_file.close()