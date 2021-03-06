from nltk.corpus.reader import PlaintextCorpusReader
from nltk.probability import FreqDist
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
from string import printable
source = "C:\\Users\\USER\\Desktop\\自然語言處理\\HW1PTTdataCrawler\\stock"
p = PlaintextCorpusReader(source, fileids = ".*\.txt")
common = []
n = 200
f = FreqDist(samples=p.words())
for i, j in f.most_common(n):
    if i not in printable:
        common.append(i)
print(common)

corpus = PathLineSentences(source)
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
print(model.wv.most_similar(positive=["漲","上"], negative=['跌']))
