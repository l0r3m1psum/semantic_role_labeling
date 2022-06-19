'''
To run form the hw2 directory use:
$ rm -f ../model/vocab.txt && caffeinate -d time python3 -m hw2.stud.my_model && echo '\a'

Running from that directory is "obligatory" because of how implementation.py
imports stuff.
'''
import os, collections, json, random, sys, math

import torch

import stud.data

SEED: int = 42

# Model's hyper-parameters
NUM_HEADS: int = 8
LEMMAS_EMBED_DIM: int = 128
PRED_EMBED_DIM: int = 16
POS_EMBED_DIM: int = 16
TOT_EMBED_DIM: int = LEMMAS_EMBED_DIM+POS_EMBED_DIM+PRED_EMBED_DIM
DROPOUT_RATE: float = 0.2 # probability of removing
LSTM_HIDDEN_DIM: int   = 128
LSTM_LAYERS: int   = 3
POSITIONAL_ENCODING: bool = True

# Vocabulary's generation parameters
LEMMA_KEEP_THRESHOLD: int = 1
LEMMA_KEEP_PROBABILITY: float = 1.0

# Trainig hyper-parameters
BATCH_SIZE: int = 100
EPOCHS: int = 20
# TODO: use PATIENCE

# Files names
MODEL_FNAME: str = '../model/model.pt'
VOCAB_FNAME: str = '../model/vocab.txt'
LOSS_FNAME: str = '../loss.dat'
TRAIN_FNAME: str = '../data/EN/train.json'
DEV_FNAME: str = '../data/EN/dev.json'

OOV_LEMMA: str = '<UNK>'
PAD_LEMMA: str = '<PAD_LEMMA>'
PAD_TAG: str = '<PAD_TAG>'
PAD_ROLE: str = '<PAD_ROLE>'
PAD_PRED: str = '<PAD_PRED>'
NULL_TAG: str = '_' # used for both roles and predicates

WORDS_KEY = 'words'
LEMMAS_KEY = 'lemmas'
POS_TAGS_KEY = 'pos_tags'
PREDICATES_KEY = 'predicates'
DEPENDENCY_HEADS_KEY = 'dependency_heads'
DEPENDENCY_RELATIONS_KEY = 'dependency_relations'
ROLES_KEY = 'roles'
KEYS = [
	WORDS_KEY,
	LEMMAS_KEY,
	POS_TAGS_KEY,
	PREDICATES_KEY,
	DEPENDENCY_HEADS_KEY,
	DEPENDENCY_RELATIONS_KEY,
	ROLES_KEY,
]
assert KEYS[-1] == ROLES_KEY

device = None
index2role = None
role2index = None
index2predicate = None
predicate2index = None
index2pos_tag = None
pos_tag2index = None
index2lemma = None
lemma2index = None

def init_globals(vocab_path) -> None:
	global device, index2role, role2index, index2predicate, predicate2index, \
		index2pos_tag, pos_tag2index, index2lemma, lemma2index

	try:
		device = torch.device('mps') if torch.backends.mps.is_available() \
			else torch.device('cuda') if torch.cuda.is_available() \
			else torch.device('cpu')
	except AttributeError as e:
		device = torch.device('cpu')
	finally:
		# Saddly the mps backend is not totally usable.
		device = torch.device('cpu')

	def make_string_to_index_converter(index2any: list) -> dict:
		return {any: index for index, any in enumerate(index2any)}

	# For ease of data inspection the padding value should always be the first
	# in each list.

	index2role = [PAD_ROLE, NULL_TAG] + [s.lower() for s in stud.data.semantic_roles]
	role2index = make_string_to_index_converter(index2role)
	assert index2role[0] == PAD_ROLE

	index2predicate = [PAD_PRED, NULL_TAG] + stud.data.predicates
	predicate2index = make_string_to_index_converter(index2predicate)
	assert index2predicate[0] == PAD_PRED

	index2pos_tag = [PAD_TAG] + stud.data.pos_tags
	pos_tag2index = make_string_to_index_converter(index2pos_tag)
	assert index2pos_tag[0] == PAD_TAG

	index2lemma = []
	lemma2index = {}
	with open(vocab_path) as f:
		for index, word in enumerate(f):
			file_line_no = index+1
			word = word.rstrip()
			if not word:
				print(f'skipping empty line in {vocab_path}:{file_line_no}')
				continue
			if ' ' in word:
				raise Exception(f'space in word found {vocab_path}:{file_line_no}')
			index2lemma.append(word)
			lemma2index[word] = index
	if OOV_LEMMA not in lemma2index:
		raise Exception('the out of vocabulary token is not present in the vocabulary')
	assert OOV_LEMMA in index2lemma
	if PAD_LEMMA not in lemma2index:
		raise Exception('the padding token is not present in the vocabulary')
	assert PAD_LEMMA in index2lemma
	if NULL_TAG in lemma2index:
		raise Exception('the null tag is present in the vocabulary')
	assert NULL_TAG not in index2lemma
	if index2lemma[0] != PAD_LEMMA:
		raise Exception('The first lemma should be the padding one.')

def is_valid_sentence(sentence: dict) -> bool:
	'''Validates some of the assumption I have about the data.'''

	if any(key not in sentence for key in KEYS[:-1]):
		print('not all keys are present', file=sys.stderr)
		print(sentence.keys())
		return False

	if any(pos_tag not in stud.data.pos_tags for pos_tag in sentence[POS_TAGS_KEY]):
		print('there is some unknown POS tag', file=sys.stderr)
		print(sentence[POS_TAGS_KEY])
		return False

	if any(predicate not in stud.data.predicates + [NULL_TAG] for predicate in sentence[PREDICATES_KEY]):
		print('there is some unknown predicate', file=sys.stderr)
		print(sentence[PREDICATES_KEY])
		return False

	sentence_len: int = len(sentence[WORDS_KEY])

	if ROLES_KEY not in sentence:
		if any(predicate != NULL_TAG for predicate in sentence[PREDICATES_KEY]):
			print('there is a predicate when there are no roles')
			print(sentence[PREDICATES_KEY])
			return False
	else:
		for roles in sentence[ROLES_KEY].values():
			if any(role.upper() not in stud.data.semantic_roles + [NULL_TAG] for role in roles):
				print(f'there is an unknown semantic role', file=sys.stderr)
				print(roles)
				return False
			if len(roles) != sentence_len:
				print('not every role has the same length', file=sys.stderr)
				print(sentence_len)
				print(roles)
				return False

	if any(len(sentence[key]) != sentence_len for key in KEYS[:-1]):
		print('not everything is of the same length', file=sys.stderr)
		return False

	return True

def to_tensor(str_list, dict_mapper, default_key, device) -> torch.Tensor:
	'''Converts a list of tring to a tensor of longs'''
	return torch.as_tensor(
		[dict_mapper.get(str, dict_mapper[default_key])
		for str in str_list],
		dtype=torch.long,
		device=device
	)

def sentence_to_tensors(sentence: dict, device) -> list:
	'''Takes a sentence dictionary, converts it into a list with an element for
	each predicate in the sentence.'''

	data = []
	predicate_null_tag = predicate2index[NULL_TAG]
	sentence_has_roles = ROLES_KEY in sentence

	lemmas = to_tensor(sentence[LEMMAS_KEY], lemma2index, OOV_LEMMA, device)
	pos_tags = to_tensor(sentence[POS_TAGS_KEY], pos_tag2index, 'X', device)
	predicates = to_tensor(sentence[PREDICATES_KEY], predicate2index, NULL_TAG, device)

	for i, predicate in enumerate(predicates):
		if predicate == predicate_null_tag: continue
		# If no predicate is presente in the sentence we append nothing
		# to the data.

		tensor_with_one_pred = torch.ones(predicates.shape[0], dtype=torch.long) \
			* predicate_null_tag
		tensor_with_one_pred[i] = predicate
		assert tensor_with_one_pred.shape == predicates.shape

		# If the sentence has no roles it means that we are trying to do a
		# prediction and we need to generate the index for the roles in the
		# sentence.
		roles_for_predicate = to_tensor(sentence[ROLES_KEY][str(i)],
				role2index, NULL_TAG, device) if sentence_has_roles else None
		data.append((lemmas, pos_tags, tensor_with_one_pred, roles_for_predicate, i))

	return data

class SRLDataset(torch.utils.data.Dataset):

	def __init__(self, sentences: list) -> None:
		super().__init__()

		self.data = []
		for sentence in sentences:
			tensors = sentence_to_tensors(sentence, torch.device('cpu'))
			self.data.extend(tensors)

	def __len__(self):
		return len(self.data)
	def __getitem__(self, index: int):
		return self.data[index]

class SRLModel(torch.nn.Module):

	def __init__(
			self,
			lemma2index: dict,
			embedding_dim_lemmas: int,
			predicate2index: dict,
			embedding_dim_predicates: int,
			pos_tag2index: dict,
			embedding_dim_pos_tag: int,
			num_heads: int,
			lstm_hidden_dim: int,
			lstm_layers_num: int,
			out_features: int
			) -> None:
		super().__init__()

		self.dropout = torch.nn.Dropout(DROPOUT_RATE)

		self.lemmas_embeddings = torch.nn.Embedding(
			len(lemma2index),
			embedding_dim_lemmas,
			lemma2index[PAD_LEMMA]
		)
		self.pos_tag_embeddings = torch.nn.Embedding(
			len(pos_tag2index),
			embedding_dim_pos_tag,
			pos_tag2index[PAD_TAG]
		)
		self.predicates_embeddings = torch.nn.Embedding(
			len(predicate2index),
			embedding_dim_predicates,
			predicate2index[PAD_PRED]
		)

		self.multihead_attention = torch.nn.MultiheadAttention(
			embedding_dim_lemmas,
			num_heads,
			batch_first=True
		)

		total_embedding_dim = embedding_dim_lemmas\
			+embedding_dim_pos_tag+embedding_dim_predicates

		self.recurrent = torch.nn.LSTM(
			input_size=total_embedding_dim,
			hidden_size=lstm_hidden_dim,
			num_layers=lstm_layers_num,
			batch_first=True,
			bidirectional=True
		)
		lstm_output_dim = 2*lstm_hidden_dim # because of bidirectional LSTM
		self.classifier = torch.nn.Linear(lstm_output_dim, out_features)

		if __debug__:
			self.out_features = out_features

	def forward(
			self,
			sentences: torch.Tensor,
			pos_tags_for_sentences: torch.Tensor,
			predicates_in_sentences: torch.Tensor
			) -> torch.Tensor:
		if __debug__: batch_size, seq_len = sentences.shape
		assert batch_size <= BATCH_SIZE

		assert sentences.dtype == torch.long
		assert predicates_in_sentences.shape == sentences.shape
		assert predicates_in_sentences.dtype == sentences.dtype
		assert pos_tags_for_sentences.shape == sentences.shape
		assert pos_tags_for_sentences.dtype == sentences.dtype

		# The int function is needed when 'mps' device is used, summing boolean
		# tensors seems to be bugged.
		assert all(
			sum(int(predicate != predicate2index[NULL_TAG] \
				and predicate != predicate2index[PAD_PRED]) \
				for predicate in predicates) == 1
			for predicates in predicates_in_sentences
		), f'there shall be exactly one predicate in each sentence'

		lemmas_embeddings = self.lemmas_embeddings(sentences)
		assert lemmas_embeddings.shape == (batch_size, seq_len, LEMMAS_EMBED_DIM)
		pos_tag_embeddings = self.pos_tag_embeddings(pos_tags_for_sentences)
		assert pos_tag_embeddings.shape == (batch_size, seq_len, POS_EMBED_DIM)
		predicates_embeddings = self.predicates_embeddings(predicates_in_sentences)
		assert predicates_embeddings.shape == (batch_size, seq_len, PRED_EMBED_DIM)

		if POSITIONAL_ENCODING:
			_, seq_len = sentences.shape
			# taken from here: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
			# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
			position = torch.arange(seq_len).unsqueeze(1) # a column vector
			div_term = torch.exp(
				torch.arange(0, LEMMAS_EMBED_DIM, 2)
				* (-math.log(10000.0) / LEMMAS_EMBED_DIM)
			) # a row vector
			pe_matrix = position * div_term # outer product
			pe = torch.zeros(seq_len, LEMMAS_EMBED_DIM)
			pe[:, 0::2] = torch.sin(pe_matrix)
			pe[:, 1::2] = torch.cos(pe_matrix)
			lemmas_embeddings += pe.to(device)

		lemmas_embeddings = self.dropout(lemmas_embeddings)

		query, key, value = (lemmas_embeddings,)*3
		attention_embeddings, _ = self.multihead_attention(query, key, value,
			key_padding_mask=(sentences == lemma2index[PAD_LEMMA]).to(device),
			need_weights=False)
		assert attention_embeddings.shape == lemmas_embeddings.shape

		# Adding the information about which word is the predicate and POS
		# tagging at the end of each embedding.
		embeddings = torch.cat(
			(attention_embeddings, pos_tag_embeddings, predicates_embeddings),
			2
		)
		embeddings = self.dropout(embeddings)
		assert embeddings.shape == (batch_size, seq_len, TOT_EMBED_DIM)

		o, (h, c) = self.recurrent(embeddings)
		o = self.dropout(o)

		out = self.classifier(o)
		assert out.shape == (batch_size, seq_len, self.out_features)
		return out

def prepare_batch(batch):

	sentences = torch.nn.utils.rnn.pad_sequence(
		[tup[0] for tup in batch],
		batch_first=True,
		padding_value=lemma2index[PAD_LEMMA]
	)
	pos_tags = torch.nn.utils.rnn.pad_sequence(
		[tup[1] for tup in batch],
		batch_first=True,
		padding_value=pos_tag2index[PAD_TAG]
	)
	predicates = torch.nn.utils.rnn.pad_sequence(
		[tup[2] for tup in batch],
		batch_first=True,
		padding_value=predicate2index[PAD_PRED]
	)
	roles = torch.nn.utils.rnn.pad_sequence(
		[tup[3] for tup in batch],
		batch_first=True,
		padding_value=role2index[PAD_ROLE]
	)

	return sentences, pos_tags, predicates, roles

def main() -> int:
	# Seeding stuff
	random.seed(SEED)
	torch.manual_seed(SEED)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.use_deterministic_algorithms(True)

	# Generating the vocabulary from the lemmas in the training data
	if not os.path.exists(VOCAB_FNAME):
		print('generating the vocabulary...')
		lemmas_counter = collections.Counter()
		with open(VOCAB_FNAME, 'w') as vocab_file, \
			open(TRAIN_FNAME) as train_data_file:

			print(PAD_LEMMA, file=vocab_file)
			print(OOV_LEMMA, file=vocab_file)

			sentences = json.load(train_data_file)

			for sentence in sentences.values():
				if not is_valid_sentence(sentence):
					print('skipping invalid sentence', file=sys.stderr)
					continue
				lemmas_counter.update(sentence[LEMMAS_KEY])

			for word, num in sorted(lemmas_counter.items()):
				# NOTE: is evaluation order defined in python?
				if num > LEMMA_KEEP_THRESHOLD and \
					random.random() <= LEMMA_KEEP_PROBABILITY:
					print(word, file=vocab_file)

		del lemmas_counter, sentences, sentence, word, num, vocab_file, train_data_file
	assert dir() == [], f'{dir()}'

	init_globals(VOCAB_FNAME)
	print(f'total number of words in the vocabulary: {len(index2lemma)}')

	with open(TRAIN_FNAME) as f: sentences_dict = json.load(f)
	train_dataset = SRLDataset(sentences_dict.values())
	train_dataloader = torch.utils.data.DataLoader(
		train_dataset,
		collate_fn=prepare_batch,
		batch_size=BATCH_SIZE,
		shuffle=True
	)

	with open(DEV_FNAME) as f: sentences_dict = json.load(f)
	validation_dataset = SRLDataset(sentences_dict.values())
	validation_dataloader = torch.utils.data.DataLoader(
		validation_dataset,
		collate_fn=prepare_batch,
		batch_size=BATCH_SIZE,
		shuffle=False
	)

	del f, sentences_dict

	# NOTE: since lemma2index, predicate2index, and pos_tag2index are global
	# variables they can also be ommited from the function arguments.
	model = SRLModel(
		lemma2index,
		LEMMAS_EMBED_DIM,
		predicate2index,
		PRED_EMBED_DIM,
		pos_tag2index,
		POS_EMBED_DIM,
		NUM_HEADS,
		LSTM_HIDDEN_DIM,
		LSTM_LAYERS,
		len(index2role)
	)
	model.to(device)
	criterion = torch.nn.CrossEntropyLoss(
		weight=torch.tensor([0.2 if role == NULL_TAG else 1.0 for role in index2role]).to(device),
		ignore_index=role2index[PAD_ROLE],
		reduction='sum'
	)
	optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0)

	log_steps: int = 10
	train_loss: float = 0.0
	losses: List[Tuple[float, float]] = []

	for epoch in range(EPOCHS):
		print(f' Epoch {epoch + 1:03d}')
		epoch_loss: float = 0.0

		model.train()
		for step, batch in enumerate(train_dataloader):
			sentences, pos_tags, predicates, actual_roles = batch
			if __debug__: batch_size, seq_len = actual_roles.shape
			assert batch_size <= BATCH_SIZE
			sentences = sentences.to(device)
			pos_tags = pos_tags.to(device)
			predicates = predicates.to(device)
			actual_roles = actual_roles.to(device)
			optimizer.zero_grad()

			# The model outputs logits.
			predicted_roles = model(sentences, pos_tags, predicates)
			predicted_roles = predicted_roles.view(-1, predicted_roles.shape[-1])
			assert predicted_roles.shape == (batch_size*seq_len, len(index2role))
			actual_roles = actual_roles.view(-1)
			assert actual_roles.shape == (batch_size*seq_len,)
			loss = criterion(predicted_roles, actual_roles)
			loss.backward() # here, if we use 'mps' device, it fails...
			optimizer.step()

			epoch_loss += loss.tolist()

			if step % log_steps == log_steps - 1:
				print(f'\t[E: {epoch:2d} @ step {step:3d}] current avg loss = '
					f'{epoch_loss / (step + 1):0.4f}')

		avg_epoch_loss = epoch_loss / len(train_dataloader)
		train_loss += avg_epoch_loss

		print(f'\t[E: {epoch:2d}] train loss = {avg_epoch_loss:0.4f}')

		valid_loss = 0.0
		model.eval()
		with torch.no_grad():
			for batch in validation_dataloader:
				sentences, pos_tags, predicates, actual_roles= batch
				sentences = sentences.to(device)
				pos_tags = pos_tags.to(device)
				predicates = predicates.to(device)
				actual_roles = actual_roles.to(device)
				predictions = model(sentences, pos_tags, predicates)
				predictions = predictions.view(-1, predictions.shape[-1])
				actual_roles = actual_roles.view(-1)
				loss = criterion(predictions, actual_roles)
				valid_loss += loss.tolist()

		valid_loss /= len(validation_dataloader)

		print(f'  [E: {epoch:2d}] valid loss = {valid_loss:0.4f}')
		losses.append((avg_epoch_loss, valid_loss))

	print('... Done!')

	avg_epoch_loss = train_loss/EPOCHS

	torch.save(model.state_dict(), MODEL_FNAME)

	with open(LOSS_FNAME, 'w') as loss_file:
		print('# train development', file=loss_file)
		for avg_epoch_loss, valid_loss in losses:
			print(f'{avg_epoch_loss} {valid_loss}', file=loss_file)

	# del losses, loss_file, batch, X, Y, avg_epoch_loss, epoch, epoch_loss, \
	# 	log_steps, loss, train_loss, valid_loss

if __name__ == '__main__':
	raise SystemExit(main())
