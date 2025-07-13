import argparse
import pickle

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from definitions.base import MW_FEW_SHOT_DOMAIN_DEFINITIONS
from loaders import load_mwoz

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_faiss_db', default='multiwoz-context-db.vec')
    parser.add_argument('--model', default='sentence-transformers/all-mpnet-base-v2', help='Embedding model name; sentence-transformers/all-mpnet-base-v2 for HuggingFace')
    parser.add_argument('--database_path', default='multiwoz_database')
    parser.add_argument('--context_size', type=int, default=3)
    parser.add_argument('--embeddings', default='huggingface', help='huggingface or openai')
    parser.add_argument('--total', default=10, type=int)
    parser.add_argument('--split', default='train', type=str)
    args = parser.parse_args()

    if args.embeddings == 'huggingface':
        embeddings = HuggingFaceEmbeddings(model_name=args.model)

    available_domains = [d for d in MW_FEW_SHOT_DOMAIN_DEFINITIONS.keys() if d != 'bus']
    data_gen = load_mwoz(args.database_path, args.context_size, split=args.split, total=args.total, available_domains=available_domains, shuffle=True, only_single_domain=True)

    docs = []
    for turn in data_gen:
        doc = Document(page_content=turn['page_content'],
                       metadata=turn['metadata'])
        docs.append(doc)
    faiss_vs = FAISS.from_documents(documents=docs,
                                    embedding=embeddings)
    with open(args.output_faiss_db, 'wb') as f:
        pickle.dump(faiss_vs, f)
