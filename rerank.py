

#{1:[{docid:[score,classi]}]}

def read_result(txt_file,num):

	file = open(txt_file, 'r', encoding='ISO-8859-1')
	text = file.read()
	file.close()

	_query = num
	query_info_list = []
	data = {}
	for run in text.split('\n'):
		run_dict = {}
		if run != '':
			query,_,doc_id,cl,score,__ = run.split(' ')
			if score == '1':
				score = float(score)
				score = score*0.5
				score = str(score)
			run_dict[doc_id] = [score,cl]
			
		else:
			query = '36'
		
		if query != _query:
			#print(_query,'llll')
			data[_query] = query_info_list
			query_info_list = []


		query_info_list.append(run_dict)
			
		_query = query
	return data

data = read_result('corpus2/r2_results.txt','1')
data2 = read_result('corpus2/r2_resultsb.txt','35')

#print(data[])

def re_rnk(data):
	re_ranked = {}
	for query in data:

		#print(len(data[query]))

		query_data = data[query]

		query_ranked = []

		scores = [float(point[doc_id][0]) for point in query_data for doc_id in point]

		for i in range(1000):

			point_dict = {}

			mx = max(scores)

			if mx == 0:
				break
			
			indx = scores.index(mx)

			docid = list(query_data[indx].keys())[0]
			score = list(query_data[indx].values())[0][0]
			classification = list(query_data[indx].values())[0][1]

			rank = i+1

			point_dict['doc_id'] = docid
			point_dict['score'] = score
			point_dict['rank'] = rank
			point_dict['classi'] = classification

			query_ranked.append(point_dict)

			scores[indx] = 0
			

		re_ranked[query] = query_ranked
	return re_ranked

re_ranked1 = re_rnk(data)
re_ranked2 = re_rnk(data2)
file = open('corpus2/final_results.txt', 'w+',encoding='utf-8')

for query in re_ranked1:

	for point_dict in re_ranked1[query]:

		doc_id = point_dict['doc_id']
		score = point_dict['score']
		rank = point_dict['rank']
		cl = point_dict['classi']

		text = str(query)+' '+'Q0'+' '+doc_id+' '+str(rank)+' '+str(score)+' '+'random_bert_tiab'

		file.write(text+"\n")

for query in re_ranked2:

	for point_dict in re_ranked2[query]:

		doc_id = point_dict['doc_id']
		score = point_dict['score']
		rank = point_dict['rank']
		cl = point_dict['classi']

		text = str(query)+' '+'Q0'+' '+doc_id+' '+str(rank)+' '+str(score)+' '+'random_bert_tiab'

		file.write(text+"\n")

file.close()







